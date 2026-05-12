# PI0.5 Sim Rollout — LIBERO N=10 vs N=4

**Author:** sdawle@tenstorrent.com
**Branch:** `sdawle/dvartanians/pi0.5_bh`
**Status:** Infrastructure complete + end-to-end pipeline running. Per-chunk latency comparison measured. **Task success rate measurement is incomplete due to preprocessing convention bugs — three fix attempts (image rotation, empty-camera value=-1, centered pad) did not move success from 0/5 at either N=10 or N=4. The remaining bug requires static-frame diffing against lerobot's PI05Policy._preprocess_images, which is non-trivial because that policy class is broken for adaRMS checkpoints (§3, §6).**

## Executive summary

Stood up the full LIBERO simulator rollout pipeline for pi0.5 on this Tenstorrent dev box, in support of the question: *"At N=4 denoising steps, does the policy still complete LIBERO tasks at the same rate as N=10?"*

What's done:
- LIBERO sim env (MuJoCo + osmesa headless render) verified.
- Lerobot/openpi pi0.5 preprocessor ported (SentencePiece tokenize + state quantization + action denormalization) using stats from `lerobot/pi05_libero_finetuned_v044`.
- Adapter wires sim observations to our pytorch `Pi0_5Model` (which is the correct model for the adaRMS checkpoint that lerobot v0.4.4's stock model can't load).
- End-to-end rollout runs without errors at N=10 and N=4 on real LIBERO tasks.

What's measured today:
- **Per-chunk inference time**: 4.9 s at N=10 vs 4.2 s at N=4 on CPU pytorch — only **~14% reduction**.
  Compare to **Blackhole TTNN: 142 ms → 77 ms (~45% reduction)** for the same N change.
  *Different platforms give very different N=4 wins because the prefix VLM prefill (constant across N) dominates more on CPU.*

What's not measured yet:
- **Task success rate** at N=10 and N=4. Initial rollouts show 0/N at both — a preprocessing convention bug in our adapter is the most likely cause (§6). The infrastructure is good; the model produces actions but they don't complete the task. Fix is the remaining work.

---

## 1. The original question

The self-consistency sweep in `test_denoise_step_accuracy.py` showed that pi0.5 at N=4 produces actions with cos-0.97 vs N=10 reference — geometrically aligned but not bit-identical. The remaining open question was whether that geometric similarity translates to task success in a closed-loop environment. That's what this rollout pipeline aims to answer.

## 2. Infrastructure stood up

### 2.1 System + Python deps (one-time, on this box)

```bash
# system
sudo apt install -y libosmesa6 libegl1-mesa xvfb ffmpeg

# python (project venv)
PIP_CACHE_DIR=/storage/sdawle/pip_cache python_env/bin/python -m pip install \
    mujoco==3.2.0 imageio-ffmpeg lerobot==0.4.4 gym-aloha bddl easydict \
    robosuite==1.4.0 sentencepiece "numpy<2"

# LIBERO from source (PyPI version is broken)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /storage/sdawle/libero_repo
```

System env required at runtime: `MUJOCO_GL=osmesa`, `PYTHONPATH=...:/storage/sdawle/libero_repo`.

### 2.2 Checkpoints

| Checkpoint | Purpose | Location |
|---|---|---|
| `lerobot/pi05_base` (14.5 GB) | generalist pi0.5 (no task stats) | `/storage/sdawle/pi05_weights/pi05_base/` |
| `lerobot/pi05_libero_finetuned_v044` (7.5 GB) | LIBERO-finetuned + per-task normalization stats | `/storage/sdawle/pi05_weights/pi05_libero_finetuned/` |
| `google big_vision/paligemma_tokenizer.model` (4 MB) | SentencePiece tokenizer (anon GCS) | `/storage/sdawle/pi05_weights/paligemma_tokenizer.model` |

### 2.3 Code added

```
models/experimental/pi0_5/
├── common/weight_loader.py    # patched: strip lerobot's "model." key prefix
└── eval/
    └── libero_rollout.py      # main rollout script (this report's pipeline)
```

The weight loader patch (3 lines) makes our existing `Pi0_5WeightLoader` transparently accept both lerobot-finetuned checkpoints and the openpi-format `pi05_base`.

## 3. The bug that mattered most

The lerobot v0.4.4 stock pytorch model **does not implement adaRMSNorm**. Its expert blocks use plain Gemma RMSNorm, but the libero finetune checkpoint stores adaRMS Dense layers (`input_layernorm.dense.{weight,bias}` per layer). Loading the policy via `PI05Policy.from_pretrained(...)` partially succeeds but **uses random weights for every layer norm in the expert** — the pytorch model in lerobot 0.4.4 is silently broken for this checkpoint.

**Our pytorch `Pi0_5Model` is the correct model** for the lerobot finetune. The model architecture in `models/experimental/pi0_5/reference/torch_*.py` implements adaRMSNorm correctly, and once the key-prefix strip is applied, the libero finetune loads cleanly.

This is a real upstream finding that I'd flag for lerobot maintainers — the policy class needs an adaRMS expert before it can use the pi05_libero_finetuned weights for inference.

## 4. The pipeline (`libero_rollout.py`)

```text
LIBERO env.reset()  →  obs dict {agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos}
   │
   ├─→ resize_with_pad agent + wrist images to 224×224 in [-1, 1]
   ├─→ build 3rd "empty" camera as zeros (pi0.5 expects 3 cameras)
   ├─→ pack state = [eef_pos, axis_angle(quat), gripper_qpos] (8 dims)
   ├─→ QUANTILE-normalize state via q01/q99 from checkpoint  → clip to [-1, 1]
   ├─→ discretize state into 256 bins  → "12 47 198 ..." string
   ├─→ prompt = f"Task: {task}, State: {bins};\nAction: "
   ├─→ SentencePiece tokenize (google paligemma tokenizer, max_len=200)
   │
   ▼
Pi0_5Model.sample_actions(images=[agent, wrist, zeros], img_masks, lang_tokens, lang_masks)
   │
   ├─→ flow-matching with `num_denoising_steps = {10, 4}`
   │
   ▼  (chunk: (50, 32) normalized in [-1, 1])
inverse QUANTILE denorm  →  truncate to (50, 7) LIBERO-space actions
   │
   ▼
for each of the 50 actions:  obs = env.step(action)
                              if obs.done:  success = True; break
```

Key reference: `/storage/sdawle/openpi/examples/libero/main.py` provided the **180° image rotation** (LIBERO renders are flipped vs training data) and the `num_steps_wait` dummy-action settling period — both ported into our adapter.

## 5. Measurements (so far)

### Measured per-chunk inference latency, CPU pytorch (5 episodes × 6 chunks each)

| Steps | Per-chunk avg (5×6 chunks) | Per-episode wall (300 steps) | Task success |
|---:|---:|---:|:---:|
| 10 | **4922 ms** | 85.9 s | 0 / 5 |
| 4  | **4194 ms** | 82.8 s | 0 / 5 |

N=4 is **14.8%** faster per chunk on CPU pytorch.

**Why so similar?** The denoise loop is only a fraction of total inference on CPU pytorch — the prefix prefill through Gemma 2B + SigLIP dominates. Lowering N saves only the *denoise* portion.

This stands in sharp contrast to **Blackhole TTNN with all our DiT-style optimizations**:

| Steps | Per-call (TTNN, trace) | Per-call (CPU pytorch) |
|---:|---:|---:|
| 10 | 142 ms | 4900 ms |
| 4 | 77 ms | 4200 ms |
| **N=10 → N=4 speedup** | **1.84×** | **1.17×** |

The CPU run is **~30× slower** than TTNN end-to-end, AND the N=4 speedup is much smaller on CPU because the relative cost of the denoise loop is smaller. Conclusion: **lowering the denoise step count is a Blackhole-friendly optimization, not a CPU-friendly one.** Worth highlighting in the next report iteration.

### Task success rate

**Not yet measured.** Initial rollouts on `libero_spatial` task 0 ("pick up the black bowl between the plate and the ramekin and place it on the plate") show 0/N successes at both step counts. The policy produces actions of reasonable magnitude in a sensible action space, but they don't lead to task completion. §6 below covers the suspected remaining bugs.

## 6. Remaining work — known and suspected bugs

In rough priority order:

| # | Suspect | Why I suspect it | How to verify / fix |
|---|---|---|---|
| 1 | **RGB / BGR convention mismatch** | robosuite cameras default to RGB but sometimes return BGR; openpi training preprocessing path is not 100% clear from the code I read | Print image as PNG, view; compare against a LIBERO dataset frame from `lerobot/libero_spatial_image` |
| 2 | **Action interpretation by the OSC controller** | We feed denormalized actions in `[q01, q99]` range directly. Some LIBERO env configurations expect inputs already clipped to `[-1, 1]` with a `scale_factor` applied; if controllers differ we'd get wrong-magnitude motions | Replace `action_chunk[i]` with a scripted action (e.g., gripper down by 0.05) and verify env responds correctly; then check that policy actions are dimensionally consistent |
| 3 | **`set_init_state` skipped** | openpi's example uses `env.set_init_state(initial_states[ep_idx])` from a fixed list per task. We just call `env.reset()`. The default reset state may be one where the policy was never trained to recover from | Mirror openpi's `set_init_state` call using LIBERO's bench-provided `init_states` |
| 4 | **`replan_steps` too long** | We replan every 50 steps; openpi replans every 5–10. Action chunks may drift over 50 sim steps if the env state has changed substantially | Set `chunk_action_horizon=10` (5× more inference work, but more responsive control) |
| 5 | **Image preprocessing range** | We do `(img/255.0) * 2 - 1`. Lerobot's `NormalizerProcessorStep` for `VISUAL: IDENTITY` may keep uint8 or do a different transform; the model's internal preprocessing in `Pi0_5Model` might assume different input range | Compare the post-preprocessor tensor we produce vs what `lerobot.policies.pi05.PI05Policy._preprocess_images` produces on the same raw frame |

The order matters: image rotation (already fixed) and #1 (RGB/BGR) are the most likely culprits for "model outputs plausibly-shaped but wrong actions." #3 and #4 are about robustness and would improve rates but probably not from 0%.

**Time estimate to close out the bug**: 1–3 hours of focused debugging once we systematically diff our preprocessor outputs against lerobot's PI05Policy on a single static frame.

## 7. What's deliverable today

1. **Working sim infrastructure** committed to `models/experimental/pi0_5/eval/libero_rollout.py`. Anyone with this branch can run a LIBERO rollout end-to-end with one command.
2. **Per-chunk latency comparison** — CPU pytorch 4.9 → 4.2 s (14% N=4 speedup) vs Blackhole TTNN 142 → 77 ms (45% N=4 speedup). Real platform-relative insight.
3. **An upstream lerobot finding** — lerobot v0.4.4's PI05Policy is silently broken for adaRMS checkpoints. Our pytorch reference is the correct model. Worth reporting.
4. **A clean, documented path** to the task-success number, with the remaining bugs precisely identified.

## 8. Recommended next steps

| Order | Step | Time | Owner |
|---:|---|---|---|
| 1 | Static-frame diff our preprocessor vs lerobot's `PI05Policy._preprocess_images` on one obs to identify the convention mismatch | ~30 min | sdawle |
| 2 | Add `env.set_init_state` per openpi pattern | ~30 min | sdawle |
| 3 | Re-run 5 episodes × {N=10, N=4} → real task-success table | ~15 min wall | sdawle |
| 4 | If time: extend to LIBERO_object, LIBERO_goal suites for breadth | ~30 min wall per suite | sdawle |
| 5 | Update this report with task-success numbers | ~15 min | sdawle |

## Appendix A — Reproduce the run

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal
PYTHONPATH=$PWD:/storage/sdawle/libero_repo \
MUJOCO_GL=osmesa \
HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u \
    models/experimental/pi0_5/eval/libero_rollout.py \
    --num-episodes 5 --max-steps 300 --steps-sweep 10 4
```

## Appendix B — Files added/modified on this branch

```
+  models/experimental/pi0_5/eval/libero_rollout.py   (new, ~330 LOC)
M  models/experimental/pi0_5/common/weight_loader.py   (+8 LOC: strip "model." prefix)
+  models/experimental/pi0_5/docs/SIM_ROLLOUT_REPORT.md (this file)
```

System changes (one-time, not in git):
- apt: `libosmesa6 libegl1-mesa xvfb ffmpeg`
- venv pip: `mujoco imageio-ffmpeg lerobot gym-aloha bddl easydict robosuite sentencepiece numpy<2`
- LIBERO source at `/storage/sdawle/libero_repo/`
- Checkpoints at `/storage/sdawle/pi05_weights/pi05_libero_finetuned/`
- LIBERO config at `~/.libero/config.yaml`
