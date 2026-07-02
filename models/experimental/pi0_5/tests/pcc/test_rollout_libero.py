# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pi0.5 closed-loop rollout harness for LIBERO.

Runs the Pi0.5 model (PyTorch reference or TTNN) as the policy inside libero_10
simulation episodes. For each (task, init_state) pair we reset the env to the
canonical LIBERO initial state, then iterate:

    1. Build Pi0.5 inputs from the current obs (images + state + task text)
    2. Sample an action chunk via model.sample_actions()
    3. Execute the first K actions in the env
    4. Check success; re-plan

Per-task success rates are reported. Run this once with --backend torch and once
with --backend ttnn to compare task success rate between the two implementations
(the target is within 2–3% absolute per the validation plan).

The two backends are run as *separate* processes with matched seeds and init
states rather than interleaved, to keep the failure modes cleanly separated.

Usage:
    # PyTorch reference
    python test_rollout_libero.py --backend torch --n-inits 5 --tasks 0 1 2

    # TTNN
    python test_rollout_libero.py --backend ttnn  --n-inits 5 --tasks 0 1 2

Env setup:
    export PYTHONPATH=<pi05>/ttnn:<pi05>
    export TT_METAL_HOME=<pi05>
    export MUJOCO_GL=egl
    export HF_TOKEN=...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

_RENDER = "--render" in sys.argv
if _RENDER:
    # Main GL backend is GLFW so we can open an on-screen viewer; robosuite's
    # offscreen renderer will still initialize EGL on its own for image obs.
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


# ---------- constants ----------


_REPO_ROOT = Path(__file__).resolve().parents[5]  # .../pcc/.../models/../tt-metal
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(_REPO_ROOT))
WEIGHTS_DIR = os.path.join(TT_METAL_HOME, "models/experimental/pi0_5/weights")
IMAGE_SIZE = 224
DEFAULT_CHUNK = 10  # responsive replanning; reproduces v044's 4/5 (chunk 50 drifts open-loop -> 1/5)
DEFAULT_MAX_STEPS = 400
DEFAULT_N_INITS = 5
TOKENIZER_MAX_LEN = 224  # Pi0.5 trained with L=200, pad to 224 for tile alignment (2*256 + 224 = 736 = 23*32)
BASE_SEED = 42
ACTION_DIM_LIBERO = 7
STATE_DIM_LIBERO = 8  # 7 joint + 1 gripper


# ---------- Pi0.5 config for LIBERO ----------


def create_pi05_config() -> PI0ModelConfig:
    config = PI0ModelConfig(
        action_dim=32,
        action_horizon=50,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
    )
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=IMAGE_SIZE,
        patch_size=14,
    )
    return config


# ---------- normalization ----------


def load_norm_stats() -> dict:
    """Fetch QUANTILES normalization stats from HuggingFaceVLA/libero/meta/stats.json.

    lerobot pi05 uses `NormalizationMode.QUANTILES` for both STATE and ACTION
    (see lerobot/policies/pi05/configuration_pi05.py `normalization_mapping`).
    The forward transform maps [q01, q99] → [-1, 1] via
        `2 * (x - q01) / (q99 - q01) - 1`
    and the inverse maps [-1, 1] → [q01, q99] via
        `(y + 1) * (q99 - q01) / 2 + q01`
    (see lerobot/processor/normalize_processor.py:362-377).
    """
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("HuggingFaceVLA/libero", "meta/stats.json", repo_type="dataset")
    with open(p) as f:
        d = json.load(f)
    return {
        "action_q01": torch.tensor(d["action"]["q01"], dtype=torch.float32),
        "action_q99": torch.tensor(d["action"]["q99"], dtype=torch.float32),
        "state_q01": torch.tensor(d["observation.state"]["q01"], dtype=torch.float32),
        "state_q99": torch.tensor(d["observation.state"]["q99"], dtype=torch.float32),
    }


def normalize_quantiles(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Forward QUANTILES normalization: [q01, q99] → [-1, 1]."""
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.tensor(eps, dtype=denom.dtype), denom)
    return 2.0 * (x - q01) / denom - 1.0


def denormalize_quantiles(
    x_norm: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Inverse QUANTILES normalization: [-1, 1] → [q01, q99]."""
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.tensor(eps, dtype=denom.dtype), denom)
    return (x_norm + 1.0) * denom / 2.0 + q01


# ---------- obs → Pi0.5 input adapter ----------


def preprocess_image_np(img: np.ndarray) -> torch.Tensor:
    """Resize to 224, apply 180° flip to match lerobot's `LiberoProcessorStep`,
    and normalize to [-1, 1].

    lerobot's `LiberoProcessorStep` (lerobot/processor/env_processor.py:59) does
    `torch.flip(img, dims=[2,3])` on env images before the policy sees them, as
    part of the env preprocessor pipeline installed by `lerobot.envs.factory`
    for LIBERO. Training loads HF parquet images directly without this flip,
    so the parquet orientation IS the training orientation. To match it at
    rollout, we must apply the same 180° flip to raw robosuite output.
    """
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = img[::-1, ::-1]  # 180° flip — matches lerobot LiberoProcessorStep
    pil = Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t - 0.5) / 0.5  # [-1, 1]


def extract_libero_state8(obs: dict) -> torch.Tensor:
    """Build the 8-dim state vector the Pi0.5-libero fine-tune was trained on.

    The HuggingFaceVLA/libero dataset stores state as:
        [eef_pos (3), eef_axisangle (3), gripper_qpos (2)]

    NOT Euler angles — the orientation is axis-angle (rotation vector), produced
    via robosuite.utils.transform_utils.quat2axisangle. First HF row:
        [-0.053, 0.007, 0.678, 3.141, 0.002, -0.090, 0.039, -0.039]
    The 3.141 in dim 3 is the x-component of the axis-angle vector for a
    "gripper pointing down" starting pose.
    """
    from robosuite.utils import transform_utils as T

    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    axisangle = T.quat2axisangle(quat_xyzw).astype(np.float32)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    if grip.size < 2:
        grip = np.concatenate([grip, np.zeros(2 - grip.size, dtype=np.float32)])
    else:
        grip = grip[:2]
    state = np.concatenate([pos, axisangle, grip])
    return torch.tensor(state[:STATE_DIM_LIBERO], dtype=torch.float32)


def pad_state_to_32(state8: torch.Tensor) -> torch.Tensor:
    """Right-pad 8-dim state to 32 with zeros and add batch dim."""
    padded = torch.zeros(32, dtype=torch.float32)
    padded[: state8.numel()] = state8
    return padded.unsqueeze(0)


def _discretize_state_for_prompt(state32_norm: torch.Tensor) -> list:
    """Match lerobot.policies.pi05.processor_pi05.Pi05PrepareStateTokenizerProcessorStep:
    np.digitize(normalized_state, np.linspace(-1, 1, 257)[:-1]) - 1
    → integers in [0, 255] for each of the 32 padded state dims.
    """
    s = state32_norm.detach().cpu().numpy()
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    return (np.digitize(s, bins) - 1).tolist()


def build_pi0_prompt(task_text: str, state32_norm: torch.Tensor) -> str:
    """Pi0.5 prompt format: "Task: ..., State: b0 b1 ... b31;\nAction: "."""
    cleaned = task_text.strip().replace("_", " ").replace("\n", " ")
    bins = _discretize_state_for_prompt(state32_norm.squeeze(0))
    state_str = " ".join(str(b) for b in bins)
    return f"Task: {cleaned}, State: {state_str};\nAction: "


def build_pi0_inputs(obs: dict, task_text: str, tokenizer, stats: dict):
    """Return (img1, img2, img_masks, lang_tokens, lang_masks, state) for Pi0.5."""
    img1 = preprocess_image_np(obs["agentview_image"])
    img2 = preprocess_image_np(obs["robot0_eye_in_hand_image"])

    state8 = extract_libero_state8(obs)
    state8_norm = normalize_quantiles(state8, stats["state_q01"], stats["state_q99"])
    state_padded = pad_state_to_32(state8_norm)

    # Pi0.5 embeds the discretized (padded, normalized) state into the prompt.
    prompt = build_pi0_prompt(task_text, state_padded)
    tok = tokenizer(
        prompt,
        padding="max_length",
        max_length=TOKENIZER_MAX_LEN,
        truncation=True,
        padding_side="right",
        return_tensors="pt",
    )
    lang_tokens = tok.input_ids.to(torch.long)
    lang_masks = tok.attention_mask.bool()

    img_masks = [
        torch.tensor([True], dtype=torch.bool),
        torch.tensor([True], dtype=torch.bool),
    ]
    return img1, img2, img_masks, lang_tokens, lang_masks, state_padded


def actions_to_env(actions_32: torch.Tensor, chunk: int, stats: dict) -> np.ndarray:
    """Take first `chunk` actions, slice to first 7 dims, denormalize."""
    a_norm = actions_32[0, :chunk, :ACTION_DIM_LIBERO]
    a = denormalize_quantiles(a_norm, stats["action_q01"], stats["action_q99"])
    return a.detach().cpu().float().numpy()


# ---------- backends ----------


class TorchBackend:
    def __init__(self, weight_loader):
        # NB: AdaRMS chunk order in tt-metal's torch_gemma is (scale, shift, gate),
        # which an empirical L2 sweep against HF training GT confirmed is correct
        # (L2=0.13 vs >1.3 for any other permutation). Don't swap.
        #
        # The prefix attention-mask + position_ids + expert-side prefix_offset
        # plumbing that used to live here as a monkey-patch is now baked into
        # torch_pi0_model.PI0Model.forward_inference / _denoise_forward.

        from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch

        self.model = PI0ModelTorch(create_pi05_config(), weight_loader)

    def sample(self, img1, img2, img_masks, lang_tokens, lang_masks, state, x0):
        saved = self.model.denoising.sample_noise
        self.model.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: x0.clone()
        try:
            with torch.no_grad():
                actions = self.model.forward_inference(
                    images=[img1, img2],
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    state=state,
                )
        finally:
            self.model.denoising.sample_noise = saved
        return actions

    def close(self):
        pass


class TTNNBackend:
    def __init__(self, weight_loader):
        import ttnn

        from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

        self.ttnn = ttnn
        self.device = ttnn.open_device(device_id=0, l1_small_size=24576)
        torch.manual_seed(BASE_SEED)
        self.model = PI0ModelTTNN(create_pi05_config(), weight_loader, self.device)

    def sample(self, img1, img2, img_masks, lang_tokens, lang_masks, state, x0):
        ttnn = self.ttnn

        # Pin x_0 for the denoising loop
        self.model.x_t_ttnn = ttnn.from_torch(
            x0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        images_ttnn = [
            ttnn.from_torch(
                im,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for im in (img1, img2)
        ]
        lang_tokens_ttnn = ttnn.from_torch(
            lang_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        with torch.no_grad():
            actions = self.model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )
        if isinstance(actions, ttnn.Tensor):
            actions = ttnn.to_torch(actions)
        return actions

    def close(self):
        self.ttnn.close_device(self.device)


# ---------- rollout loop ----------


_LIBERO_DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def rollout_episode(env, task_text, init_state, backend, stats, tokenizer, chunk, max_steps, x0, viewer=None):
    """Run a single episode. Returns (success, steps_executed, inference_times_ms).

    Matches lerobot's LiberoEnv.reset behavior: 10 warm-up no-op steps with
    dummy action [0,0,0,0,0,0,-1] (gripper closed) after set_init_state, then
    begin policy rollout.

    IMPORTANT: env.reset() replaces sim.data._data with a new MjData object, which
    invalidates any mujoco.viewer bound to the prior pointer. The caller is
    responsible for (re-)launching the viewer AFTER each reset. This function
    assumes reset() has already been called and skips it.
    """
    obs = env.set_init_state(init_state)
    if viewer is not None:
        viewer.sync()
    # Warm-up steps (matches lerobot's num_steps_wait=10)
    for _ in range(10):
        obs, _, _, _ = env.step(_LIBERO_DUMMY_ACTION)
        if viewer is not None:
            # Force robosuite's sim to flush state to the underlying MjData
            # before the viewer reads it, otherwise the viewer sees stale qpos.
            env.env.sim.forward()
            with viewer.lock():
                viewer.sync()
            time.sleep(0.01)

    success = False
    steps = 0
    inference_ms = []

    while steps < max_steps and not success:
        pi_in = build_pi0_inputs(obs, task_text, tokenizer, stats)
        t0 = time.time()
        actions_32 = backend.sample(*pi_in, x0)
        inference_ms.append((time.time() - t0) * 1000)

        actions_np = actions_to_env(actions_32, chunk, stats)
        for k in range(actions_np.shape[0]):
            if steps >= max_steps:
                break
            obs, reward, done, info = env.step(actions_np[k])
            steps += 1
            if viewer is not None:
                viewer.sync()
            if env.check_success():
                success = True
                break
            if done:
                break

    return success, steps, inference_ms


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "ttnn"], required=True)
    parser.add_argument(
        "--weights",
        default="v044",
        help="weights dir under models/experimental/pi0_5/weights/. The LIBERO 4/5 benchmark uses 'v044' "
        "(lerobot/pi05_libero_finetuned_quantiles_v044, model.-prefix stripped); base 'pi05_libero' is weaker (~2/5).",
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", type=int, nargs="+", default=None, help="task indices (default: all)")
    parser.add_argument("--n-inits", type=int, default=DEFAULT_N_INITS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--out", default=None, help="path to write per-episode JSON results")
    parser.add_argument(
        "--render", action="store_true", help="open on-screen MuJoCo viewer during rollout (requires DISPLAY)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"  PI0.5 CLOSED-LOOP LIBERO ROLLOUT  —  backend={args.backend}")
    print("=" * 80)

    # Late imports so TTNN failures don't block torch path
    from transformers import AutoTokenizer
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    def make_env(bddl_path):
        """Always headless offscreen for policy image obs. On-screen viz goes through
        mujoco.viewer.launch_passive attached to the same sim data."""
        return OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=256,  # lerobot's LiberoEnv captures at 256 then resizes to 224 in policy preproc
            camera_widths=256,
        )

    def launch_viewer(env):
        """Return a mujoco.viewer handle bound to robosuite's sim, or None on failure.

        We have to dance around three quirks:
          1. robosuite's MjSim.data._data is the real mujoco.MjData, but its
             internal physics step uses a SEPARATE renderer context (EGL) that
             doesn't notify the GLFW viewer of state changes.
          2. mujoco.viewer.launch_passive runs an off-thread render loop, so we
             must call viewer.sync() to push the latest state to it.
          3. The viewer starts in paused mode by default — flip it off so the
             camera tracks the live data.
        """
        try:
            import mujoco.viewer as mj_viewer

            model = env.env.sim.model._model
            data = env.env.sim.data._data
            v = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
            # Force the viewer's tracked data pointer to point at robosuite's
            # current MjData object so subsequent sim.forward() calls update it.
            try:
                v._user_scn  # touch attr to ensure handle is alive
            except Exception as e:  # attribute may not exist on all viewer versions
                print(f"Viewer probe failed (non-fatal): {e}", file=sys.stderr)
            # Start synchronized
            v.sync()
            print(f"   🖥️  MuJoCo viewer opened on DISPLAY={os.environ.get('DISPLAY')}")
            print(
                f"       model={type(model).__module__}.{type(model).__name__}, "
                f"data={type(data).__module__}.{type(data).__name__}"
            )
            return v
        except Exception as e:
            print(f"   ⚠️ viewer launch failed: {e}")
            return None

    ckpt = os.path.join(WEIGHTS_DIR, args.weights)
    if not os.path.exists(ckpt):
        print(f"❌ weights dir not found: {ckpt}")
        return 1
    print(f"📁 weights: {ckpt}")
    print(f"📋 suite={args.suite}, n_inits={args.n_inits}, max_steps={args.max_steps}, chunk={args.chunk}")

    print("\n1. Loading normalization stats...")
    stats = load_norm_stats()
    print(f"   state_q01[:3]={stats['state_q01'][:3].tolist()}")
    print(f"   action_q01[:3]={stats['action_q01'][:3].tolist()}")

    print("\n2. Loading PaliGemma tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    print("\n3. Loading task suite...")
    bm = benchmark.get_benchmark_dict()[args.suite]()
    n_total_tasks = bm.n_tasks
    task_indices = args.tasks if args.tasks is not None else list(range(n_total_tasks))
    print(f"   {len(task_indices)} / {n_total_tasks} tasks selected")

    print("\n4. Loading weights...")
    weight_loader = PI0WeightLoader(ckpt)

    print(f"\n5. Initializing backend={args.backend}...")
    if args.backend == "torch":
        backend = TorchBackend(weight_loader)
    else:
        backend = TTNNBackend(weight_loader)

    # Fixed initial flow-matching noise, same for every episode and every backend
    torch.manual_seed(args.seed)
    x0 = torch.randn(1, 50, 32, dtype=torch.float32)

    results = []  # list of dicts: task_idx, init_idx, success, steps, inference_ms_mean

    try:
        for task_idx in task_indices:
            task = bm.get_task(task_idx)
            task_text = task.language
            bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            # Load init states directly — LIBERO's bm.get_task_init_states() uses
            # torch.load() with the torch 2.6 default weights_only=True, which
            # rejects the numpy reconstructor in these pickles.
            init_states_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
            init_states = torch.load(init_states_path, weights_only=False)
            if hasattr(init_states, "cpu"):
                init_states = init_states.cpu().numpy()
            elif not isinstance(init_states, np.ndarray):
                init_states = np.asarray(init_states)
            n_avail = init_states.shape[0]
            n_use = min(args.n_inits, n_avail)
            print(f"\n=== task {task_idx}: {task_text} ===")
            print(f"    n_init_states available={n_avail}, using={n_use}")

            env = make_env(bddl_path)
            # env.reset() replaces sim.data._data with a new MjData object, so
            # any mujoco.viewer bound to the prior pointer is frozen. We re-launch
            # the viewer per episode AFTER each reset inside rollout_episode, and
            # close the previous one at the top of each iteration.
            viewer = None
            try:
                for init_idx in range(n_use):
                    # Reset, then (re-)launch viewer bound to the fresh MjData pointer
                    env.reset()
                    if viewer is not None:
                        try:
                            viewer.close()
                        except Exception as e:  # ignore errors during viewer cleanup
                            print(f"Viewer close failed (non-fatal): {e}", file=sys.stderr)
                        viewer = None
                    if args.render:
                        viewer = launch_viewer(env)

                    t_start = time.time()
                    success, steps, inf_ms = rollout_episode(
                        env,
                        task_text,
                        init_states[init_idx],
                        backend,
                        stats,
                        tokenizer,
                        args.chunk,
                        args.max_steps,
                        x0,
                        viewer=viewer,
                    )
                    ep_time = time.time() - t_start
                    mean_inf = np.mean(inf_ms) if inf_ms else 0.0
                    print(
                        f"    init {init_idx:>2}: success={str(success):>5}  steps={steps:>3}  "
                        f"ep_time={ep_time:6.1f}s  mean_inf_ms={mean_inf:6.1f}  n_replans={len(inf_ms)}"
                    )
                    results.append(
                        {
                            "task_idx": task_idx,
                            "task": task_text,
                            "init_idx": init_idx,
                            "success": bool(success),
                            "steps": int(steps),
                            "ep_time_s": float(ep_time),
                            "mean_inference_ms": float(mean_inf),
                            "n_replans": len(inf_ms),
                        }
                    )
            finally:
                if viewer is not None:
                    try:
                        viewer.close()
                    except Exception as e:  # ignore errors during viewer cleanup
                        print(f"Viewer close failed (non-fatal): {e}", file=sys.stderr)
                env.close()

        # Aggregate
        print("\n" + "=" * 80)
        print("  SUMMARY")
        print("=" * 80)
        by_task = {}
        for r in results:
            by_task.setdefault(r["task_idx"], []).append(r)
        total_succ = 0
        total_run = 0
        for tidx in sorted(by_task.keys()):
            rs = by_task[tidx]
            s = sum(1 for r in rs if r["success"])
            n = len(rs)
            total_succ += s
            total_run += n
            task_text = rs[0]["task"]
            print(f"  task {tidx:>2}: {s}/{n}  ({100.0*s/n:5.1f}%)  {task_text[:60]}")
        if total_run > 0:
            print(f"\n  OVERALL: {total_succ}/{total_run}  ({100.0*total_succ/total_run:.2f}%)")
        print("=" * 80)

        if args.out:
            with open(args.out, "w") as f:
                json.dump(
                    {
                        "backend": args.backend,
                        "weights": args.weights,
                        "suite": args.suite,
                        "n_inits": args.n_inits,
                        "chunk": args.chunk,
                        "max_steps": args.max_steps,
                        "seed": args.seed,
                        "results": results,
                    },
                    f,
                    indent=2,
                )
            print(f"\n📝 wrote {args.out}")

        return 0

    except Exception as e:
        import traceback

        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
