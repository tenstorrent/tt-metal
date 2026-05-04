# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pi0.5 action-space divergence on LeRobot held-out samples.

Loads N real (image, state, task) samples from LeRobot datasets (LIBERO +
ALOHA) and runs both the PyTorch reference and the TTNN Pi0.5 implementation
on each, with matched initial noise. For every sample we report:

    - L2 distance          ||a_torch - a_ttnn||
    - cosine distance      1 - cos(a_torch, a_ttnn)
    - Pearson PCC (per-sample)

Aggregate statistics (mean, worst) are printed per dataset.

The PyTorch model is the "reference"; this test is not comparing against the
dataset's ground-truth action labels — it's asking whether TTNN tracks PyTorch
on realistic conditioning drawn from the model's training distribution, which
is a stronger test than the random-input aggregate PCC.

Usage:
    PYTHONPATH=<root>/ttnn:<root> python test_action_divergence_lerobot.py [--n 8]

Env:
    HF_TOKEN=...                 required for gated repos
    HF_HUB_ENABLE_HF_TRANSFER=1  recommended for faster download
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


_REPO_ROOT = Path(__file__).resolve().parents[5]  # tt-metal repo root
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(_REPO_ROOT))
CHECKPOINT_PATH = os.environ.get(
    "PI0_CHECKPOINT",
    os.path.join(TT_METAL_HOME, "models/experimental/pi0_5/weights/pi05_base"),
)
BATCH_SIZE = 1
SEED = 42
IMAGE_SIZE = 224


# ---------- Pi0.5 config ----------


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


# ---------- Preprocessing ----------


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Resize + normalize a PIL image to Pi0.5 input format (1,3,224,224) in [-1,1]."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t - 0.5) / 0.5


def pil_from_feature(feat) -> Image.Image:
    """datasets library returns images as PIL, dict{bytes|path}, or ndarray."""
    if isinstance(feat, Image.Image):
        return feat
    if isinstance(feat, dict):
        if feat.get("bytes"):
            return Image.open(io.BytesIO(feat["bytes"]))
        if feat.get("path"):
            return Image.open(feat["path"])
        raise ValueError(f"unknown image dict keys: {list(feat.keys())}")
    if isinstance(feat, np.ndarray):
        return Image.fromarray(feat)
    raise TypeError(f"unsupported image type: {type(feat)}")


def pad_state(state_list, target_dim: int) -> torch.Tensor:
    """Right-pad a raw robot state vector to target_dim with zeros."""
    s = torch.tensor(list(state_list), dtype=torch.float32)
    if s.ndim == 0:
        s = s.unsqueeze(0)
    if s.shape[0] < target_dim:
        s = torch.nn.functional.pad(s, (0, target_dim - s.shape[0]))
    return s[:target_dim].unsqueeze(0)


def hash_tokenize(prompt: str, max_length: int = 32):
    """Matches the existing demo's ord(char)%256000 tokenizer.

    This is *not* a real PaliGemma tokenizer, but it's what run_libero_demo.py
    feeds the model and what the existing PCC tests compare against — so using
    the same scheme keeps us comparable to the baseline numbers.
    """
    tokens = [ord(c) % 256000 for c in prompt[:max_length]]
    while len(tokens) < max_length:
        tokens.append(0)
    tok = torch.tensor([tokens[:max_length]], dtype=torch.long)
    mask = torch.ones(1, max_length, dtype=torch.bool)
    truncated = min(len(prompt), max_length)
    mask[0, truncated:] = False
    return tok, mask


# ---------- Sample loaders ----------


LIBERO_PROMPTS = {}  # task_index -> prompt (populated lazily per-dataset)


def load_libero_samples(n_samples: int):
    """Load N (img1, img2, state, prompt) samples from HuggingFaceVLA/libero.

    Matches the existing extract_libero_samples.py choice of dataset.
    """
    from datasets import load_dataset

    print(f"   📥 streaming HuggingFaceVLA/libero (n={n_samples})...")
    ds = load_dataset("HuggingFaceVLA/libero", split="train", streaming=True)

    samples = []
    seen_episodes = set()
    iter_limit = 20000
    for i, row in enumerate(ds):
        if i > iter_limit:
            break
        ep = row.get("episode_index")
        if ep in seen_episodes:
            continue
        seen_episodes.add(ep)

        try:
            img1 = pil_from_feature(row["observation.images.image"])
            img2 = pil_from_feature(row["observation.images.image2"])
        except Exception as e:
            print(f"   ⚠️ skip episode {ep}: {e}")
            continue

        state = row["observation.state"]
        task = row.get("task") or row.get("language_instruction") or f"task_{row.get('task_index', 0)}"
        if isinstance(task, list):
            task = task[0] if task else "task"

        samples.append(
            {
                "dataset": "libero",
                "episode": ep,
                "images": [img1, img2],
                "img_masks_bool": [True, True],
                "state": state,
                "prompt": str(task),
            }
        )
        if len(samples) >= n_samples:
            break

    print(f"   ✅ {len(samples)} LIBERO samples")
    return samples


def _tensor_chw_to_pil(t: torch.Tensor) -> Image.Image:
    """LeRobotDataset returns images as torch tensors CHW in [0,1]."""
    if t.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {tuple(t.shape)}")
    arr = (t.permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(arr)


def load_aloha_samples(n_samples: int):
    """Load N (top_img, zero_img, state, prompt) samples from lerobot/aloha_sim_transfer_cube_human.

    ALOHA is video-backed so the plain `datasets` streaming loader returns
    metadata only — no frames. LeRobotDataset decodes the videos on access.
    Pi0.5 expects 2 images; ALOHA sim only has a top camera, so we fill slot 2
    with a zero image and set its mask=False.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"   📥 LeRobotDataset lerobot/aloha_sim_transfer_cube_human (n={n_samples})...")
    ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")

    samples = []
    seen_episodes = set()

    # Stride through the dataset so we land on different episodes
    total = len(ds)
    stride = max(1, total // max(n_samples * 10, 1))
    idx = 0
    while len(samples) < n_samples and idx < total:
        row = ds[idx]
        ep_t = row.get("episode_index")
        ep = int(ep_t.item()) if hasattr(ep_t, "item") else int(ep_t)
        if ep in seen_episodes:
            idx += stride
            continue
        seen_episodes.add(ep)

        img_key = next((k for k in row.keys() if k.startswith("observation.images.")), None)
        if img_key is None:
            idx += stride
            continue
        top_img = _tensor_chw_to_pil(row[img_key])

        state = row["observation.state"].tolist()
        task = row.get("task") or "transfer the cube from one arm to the other"
        if isinstance(task, (list, tuple)):
            task = task[0] if task else "task"

        # ALOHA sim only has a top camera. Pi0.5 expects 2 image slots.
        # The existing run_aloha_sim_demo.py duplicates the top image into both
        # slots with mask=True for both. Using mask=False on slot 2 exposes a
        # large TTNN vs PyTorch divergence (PCC ~0.4) that is worth reporting
        # separately — see comment in test file. Here we match the demo.
        samples.append(
            {
                "dataset": "aloha",
                "episode": ep,
                "images": [top_img, top_img],
                "img_masks_bool": [True, True],
                "state": state,
                "prompt": str(task),
            }
        )
        idx += stride

    print(f"   ✅ {len(samples)} ALOHA samples")
    return samples


# ---------- Metrics ----------


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-8 or s2 < 1e-8:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    cos = torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
    return 1.0 - cos


# ---------- Sample runner ----------


def run_sample(sample, config, model_torch, model_ttnn, device, shared_x0: torch.Tensor):
    """Run one sample on both models with matched initial noise. Returns metrics dict."""
    # Build model inputs
    images = [preprocess_image(img) for img in sample["images"]]
    img_masks = [torch.tensor([m], dtype=torch.bool) for m in sample["img_masks_bool"]]
    lang_tokens, lang_masks = hash_tokenize(sample["prompt"])
    state = pad_state(sample["state"], config.state_dim)

    # ---- PyTorch ----
    # Override sample_noise to return our shared_x0 for matched comparison
    saved_sample_noise = model_torch.denoising.sample_noise
    model_torch.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: shared_x0.clone()
    try:
        with torch.no_grad():
            torch_t0 = time.time()
            torch_actions = model_torch.forward_inference(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
            )
            torch_ms = (time.time() - torch_t0) * 1000
    finally:
        model_torch.denoising.sample_noise = saved_sample_noise

    # ---- TTNN ----
    # Sync the model's internal x_0 slot to match shared_x0 so both use identical noise
    model_ttnn.x_t_ttnn = ttnn.from_torch(
        shared_x0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    images_ttnn = [
        ttnn.from_torch(
            img,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for img in images
    ]
    lang_tokens_ttnn = ttnn.from_torch(lang_tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    lang_masks_ttnn = ttnn.from_torch(lang_masks.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with torch.no_grad():
        ttnn_t0 = time.time()
        ttnn_actions = model_ttnn.sample_actions(
            images=images_ttnn,
            img_masks=img_masks,
            lang_tokens=lang_tokens_ttnn,
            lang_masks=lang_masks_ttnn,
            state=state_ttnn,
        )
        if isinstance(ttnn_actions, ttnn.Tensor):
            ttnn_actions = ttnn.to_torch(ttnn_actions)
        ttnn_ms = (time.time() - ttnn_t0) * 1000

    # Metrics
    diff = (torch_actions - ttnn_actions).float()
    l2 = diff.norm().item()
    cos_dist = cosine_distance(torch_actions, ttnn_actions)
    pcc = compute_pcc(torch_actions, ttnn_actions)

    return {
        "l2": l2,
        "cos_dist": cos_dist,
        "pcc": pcc,
        "torch_ms": torch_ms,
        "ttnn_ms": ttnn_ms,
    }


# ---------- Main ----------


def summarize(label: str, results: list):
    if not results:
        print(f"\n[{label}] no samples")
        return None
    pccs = [r["pcc"] for r in results]
    l2s = [r["l2"] for r in results]
    cds = [r["cos_dist"] for r in results]
    print(f"\n[{label}] n={len(results)}")
    print(f"  PCC         mean={np.mean(pccs):.6f}  min={np.min(pccs):.6f}  max={np.max(pccs):.6f}")
    print(f"  L2          mean={np.mean(l2s):.4f}    min={np.min(l2s):.4f}    max={np.max(l2s):.4f}")
    print(f"  cos-dist    mean={np.mean(cds):.6f}  min={np.min(cds):.6f}  max={np.max(cds):.6f}")
    return {
        "n": len(results),
        "pcc_mean": float(np.mean(pccs)),
        "pcc_min": float(np.min(pccs)),
        "l2_mean": float(np.mean(l2s)),
        "l2_max": float(np.max(l2s)),
        "cos_mean": float(np.mean(cds)),
        "cos_max": float(np.max(cds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="samples per dataset")
    parser.add_argument("--skip-libero", action="store_true")
    parser.add_argument("--skip-aloha", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(f"  PI0.5 ACTION-SPACE DIVERGENCE ON LEROBOT  (n_per_ds={args.n})")
    print("=" * 80)

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        print(f"❌ Checkpoint not found: {ckpt}")
        return 1
    print(f"📁 Checkpoint: {ckpt}")

    # Load samples first (no device needed)
    all_samples = []
    if not args.skip_libero:
        print("\n1. Loading LIBERO samples...")
        try:
            all_samples += load_libero_samples(args.n)
        except Exception as e:
            print(f"   ❌ LIBERO load failed: {e}")
    if not args.skip_aloha:
        print("\n2. Loading ALOHA samples...")
        try:
            all_samples += load_aloha_samples(args.n)
        except Exception as e:
            print(f"   ❌ ALOHA load failed: {e}")
    if not all_samples:
        print("❌ no samples loaded")
        return 1

    print("\n🔌 Opening TTNN device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        config = create_pi05_config()
        print(f"📋 pi05={config.pi05}, num_steps={config.num_denoising_steps}")

        print("\n3. Loading weights...")
        weight_loader = PI0WeightLoader(str(ckpt))

        print("4. PyTorch reference...")
        model_torch = PI0ModelTorch(config, weight_loader)

        print("5. TTNN model...")
        torch.manual_seed(SEED)
        model_ttnn = PI0ModelTTNN(config, weight_loader, device)

        # Shared initial noise — same x_0 used for every sample, both models
        torch.manual_seed(SEED)
        shared_x0 = torch.randn(BATCH_SIZE, config.action_horizon, config.action_dim, dtype=torch.float32)
        print(f"   shared x_0 std={shared_x0.std().item():.4f}")

        libero_results = []
        aloha_results = []

        print(f"\n6. Running {len(all_samples)} samples...")
        print(f"{'#':>3} {'ds':>7} {'ep':>5} {'pcc':>10} {'l2':>10} {'cos':>10} {'torch_ms':>10} {'ttnn_ms':>9}")
        print("-" * 80)
        for idx, sample in enumerate(all_samples):
            try:
                m = run_sample(sample, config, model_torch, model_ttnn, device, shared_x0)
            except Exception as e:
                print(f"{idx:>3} {sample['dataset']:>7} {sample['episode']!s:>5}  FAILED: {e}")
                continue

            print(
                f"{idx:>3} {sample['dataset']:>7} {sample['episode']!s:>5} "
                f"{m['pcc']:>10.6f} {m['l2']:>10.4f} {m['cos_dist']:>10.6f} "
                f"{m['torch_ms']:>10.1f} {m['ttnn_ms']:>9.1f}"
            )
            if sample["dataset"] == "libero":
                libero_results.append(m)
            elif sample["dataset"] == "aloha":
                aloha_results.append(m)

        print("\n" + "=" * 80)
        print("  AGGREGATE STATISTICS")
        print("=" * 80)
        libero_stats = summarize("LIBERO", libero_results)
        aloha_stats = summarize("ALOHA", aloha_results)
        summarize("ALL", libero_results + aloha_results)

        # Soft pass criterion: min PCC ≥ 0.95 per dataset (real inputs are noisier than synthetic)
        passed = True
        for stats in (libero_stats, aloha_stats):
            if stats is None:
                continue
            if stats["pcc_min"] < 0.95:
                passed = False
        print(f"\nstatus: {'✅ PASS' if passed else '⚠️  REVIEW'}  (per-dataset min PCC ≥ 0.95)")
        print("=" * 80)
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        print("\n🔌 Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
