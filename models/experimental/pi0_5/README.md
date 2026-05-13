# PI0.5 (`pi0_5`) — Tenstorrent

End-to-end TTNN implementation of the **π₀.₅** (PI0.5) vision-language-action policy on Blackhole, with a PyTorch reference, LIBERO simulator integration, and a real-weights perf path that runs at **~96 ms / chunk** (~520 actions/s) with trace+2CQ.

This package is **self-contained** — it does not import from `models/experimental/pi0/`.

---

## Architecture

```
                           ┌──────────────────────────────────────────────────────┐
                           │                  Inputs (per chunk)                   │
                           │                                                       │
                           │  ┌────────┐  ┌────────┐  ┌──────────┐  ┌───────────┐ │
                           │  │ image1 │  │ image2 │  │ empty cam│  │ lang+state│ │
                           │  │224×224 │  │ wrist  │  │  -1 fill │  │ 200 tokens│ │
                           │  └───┬────┘  └───┬────┘  └────┬─────┘  └─────┬─────┘ │
                           └──────┼───────────┼────────────┼──────────────┼───────┘
                                  │           │            │              │
                                  ▼           ▼            ▼              ▼
                             ┌──────────────────────────────────┐  ┌────────────┐
                             │      SigLIP-27 vision tower      │  │  Embeddings│
                             │   (3× batched in single bs=N)    │  │  (sqrt-scl)│
                             └────────────────┬─────────────────┘  └─────┬──────┘
                                              │                          │
                                              └─────────────┬────────────┘
                                                            ▼
                                              ┌─────────────────────────────┐
                                              │ Prefix = [image][lang+state]│
                                              │   (~750 tokens, TILE)       │
                                              └──────────────┬──────────────┘
                                                             ▼
                                       ┌─────────────────────────────────────┐
                                       │   VLM (Gemma 2B, 18-layer, KV-cache) │
                                       │     bidirectional prefix prefill     │
                                       │     → past_key_values (KV cache)     │
                                       └────────────────────┬─────────────────┘
                                                            │
                                            ┌───────────────┴───────────────┐
                                            │                               │
                                            │       Denoise loop (Euler)    │
                                            │       for t in [1.0 → 0.0]:   │
                                            │                               │
                                            │   ┌─────────────────────┐     │
   x_t ──── action_in_proj ──► action emb ──┼──►│ AdaRMS Expert 18×   │     │
                                            │   │ Gemma 300M layers   │     │
   sincos(t) ─► time_mlp ──► adarms_cond ──►│──►│  (uses adarms_cond) │     │
                                            │   └──────────┬──────────┘     │
                                            │              │                │
                                            │              ▼                │
                                            │      action_out_proj          │
                                            │              │                │
                                            │              ▼                │
                                            │       velocity v_t            │
                                            │              │                │
                                            │       x_t ← x_t + dt·v_t     │
                                            └───────────────┬───────────────┘
                                                            ▼
                                                   actions (B, 50, 32)
                                                   → take [:, :, :7] for libero
```

### What differs from PI0

| Component        | PI0                                              | **PI0.5**                                                     |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Suffix tokens    | `[state_token, action_0, …, action_{H-1}]`       | `[action_0, …, action_{H-1}]` (state encoded into lang tokens)|
| Time injection   | concat(action, sincos(t)) → 2-layer MLP, fused   | sincos(t) → MLP → `adarms_cond` (fed to adaRMSNorm)           |
| Expert RMSNorm   | Plain RMSNorm                                    | adaRMSNorm: `normed * (1+scale) + shift`, with gated residual |
| `max_token_len`  | 48                                               | 200                                                           |
| State input      | continuous (state_proj)                          | discretized into 256 bins, embedded as language tokens        |

Everything else (SigLIP-27, Gemma-2B VLM, Gemma-300M expert, flow-matching denoising with Euler integration, KV-cache prefill of the prefix) follows the openpi/lerobot reference.

---

## Directory layout

```
pi0_5/
├── common/
│   ├── configs.py            # GemmaConfig, SigLIPConfig, PaliGemmaConfig,
│   │                         #   SuffixConfig, PrefixConfig, DenoiseConfig,
│   │                         #   PI0ModelConfig, Pi0_5ModelConfig
│   └── weight_loader.py      # PI0Config, PI0WeightLoader, categorize_weights,
│                             #   Pi0_5WeightLoader (subclass; strips "model." prefix)
├── reference/                # PyTorch reference (used for PCC + libero pytorch backend)
│   ├── torch_denoise.py
│   ├── torch_gemma.py        # GemmaAttn/MLP/Block + AdaRMSGemmaBlock
│   ├── torch_paligemma.py    # PaliGemmaBackbone + Pi0_5PaliGemmaBackbone
│   ├── torch_prefix.py
│   ├── torch_siglip.py
│   ├── torch_suffix.py       # SuffixEmbedding + Pi0_5SuffixEmbedding (sincos→silu→MLP→silu)
│   └── torch_pi0_5_model.py  # Pi0_5Model
├── tt/                       # TTNN implementation
│   ├── ttnn_common.py
│   ├── ttnn_gemma.py         # GemmaAttn/MLP TTNN + AdaRMSGemmaBlockTTNN +
│   │                         #   build_matmul_pcfg (2D + 1D width-shard small-M)
│   ├── ttnn_paligemma.py     # PaliGemmaBackboneTTNN + Pi0_5PaliGemmaBackboneTTNN
│   ├── ttnn_prefix.py
│   ├── ttnn_siglip.py
│   ├── ttnn_suffix.py        # SuffixEmbeddingTTNN + Pi0_5SuffixEmbeddingTTNN
│   └── ttnn_pi0_5_model.py   # Pi0_5ModelTTNN
├── eval/
│   └── libero_rollout.py     # LIBERO simulator → policy → success rate / videos
├── tests/
│   ├── pcc/                  # Reference-vs-spec correctness
│   └── perf/                 # Latency / throughput on Blackhole
└── weights/
    └── pi05_base/            # Symlink or directory of pi05_base safetensors
```

---

## Quickstart

### Inference (PyTorch reference)

```python
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

loader = Pi0_5WeightLoader("/path/to/pi05_base")
model  = Pi0_5Model(Pi0_5ModelConfig(), loader)

actions = model.sample_actions(
    images=[img_tensor],            # list of (B, 3, 224, 224)
    img_masks=[mask],
    lang_tokens=tok_ids,            # (B, L) — includes discretized state
    lang_masks=tok_mask,
    state=None,                     # ignored on the pi0.5 path
)  # → (B, action_horizon, action_dim)
```

### Inference (TTNN, Blackhole)

```python
import ttnn
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=80_000_000)
loader = Pi0_5WeightLoader("/path/to/pi05_base")
model  = Pi0_5ModelTTNN(Pi0_5ModelConfig(), loader, device)

actions = model.sample_actions(
    images=[image_ttnn],            # list of ttnn.Tensor
    img_masks=[mask_ttnn],
    lang_tokens=tokens_ttnn,
    lang_masks=lang_masks_ttnn,
    state=None,
)
ttnn.synchronize_device(device)
actions_torch = ttnn.to_torch(actions)
```

---

## Tests

All tests below are skipped automatically if `models/experimental/pi0_5/weights/pi05_base/model.safetensors` is missing.

### PCC (correctness) tests

```bash
# Run all PCC tests (no device required for some; the *_real_weights tests
# need a Blackhole device for the TTNN path)
PYTHONPATH=$PWD python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/

# Individual:
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_suffix.py            # suffix layer (sincos+MLP, time_mlp_out silu)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_adarms_gemma.py      # AdaRMS Gemma block
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_e2e_reference.py     # E2E reference (no device)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_real_weights.py      # E2E pytorch with real pi05_base weights
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_ttnn_real_weights.py # E2E TTNN with real weights — needs device
```

### Perf tests (Blackhole)

```bash
# Headline: full sample_actions with trace + 2CQ (the published number)
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py
#  → Per-call avg ≈ 96-103 ms (N=10),  Action throughput ≈ 487-520 actions/s

# Without trace — apples-to-apples vs untraced rollout (~200 ms / chunk)
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e.py

# Single forward pass (model only, no denoise loop)
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn.py
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_trace.py        # with trace

# Stage breakdown (separate SigLIP / VLM prefill / denoise step timings)
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_trace_e2e.py

# Self-consistency: how many denoise steps can you drop before actions
# diverge from the N=10 reference? (cos sim + max delta sweep)
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_denoise_step_accuracy.py
```

**Latest measured trace-mode perf (Blackhole, N=10, after the matmul-sharding port from `pi0_p150`):**

| metric | value |
|---|---|
| per-call latency | **96.26 ms** |
| chunk throughput | 9.74 chunks/s |
| action throughput | **519.4 actions/s** |
| jitter (stddev) | 0.04 ms |
| trace capture (one-time) | ~475 ms |

---

## LIBERO simulator rollout

End-to-end real-robot benchmark on the LIBERO suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`).

### One-time setup (not in git)

```bash
# 1. PaliGemma tokenizer (used to encode the prompt + discretized state)
mkdir -p /storage/sdawle/pi05_weights
curl -L -o /storage/sdawle/pi05_weights/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# 2. pi05_libero finetune checkpoint (or your own pi0.5 LIBERO checkpoint)
#    Expected layout: <ckpt_dir>/model.safetensors
#                     <ckpt_dir>/policy_preprocessor_step_2_normalizer_processor.safetensors

# 3. LIBERO env from source (PyPI install is broken)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /storage/sdawle/libero_repo

# 4. System packages for MuJoCo headless render
sudo apt install -y libosmesa6 libegl1-mesa xvfb ffmpeg

# 5. Python deps in the active venv
python_env/bin/pip install mujoco imageio-ffmpeg lerobot gym-aloha bddl easydict robosuite sentencepiece 'numpy<2'
```

### Running a rollout

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal

PYTHONPATH=$PWD:/storage/sdawle/libero_repo \
MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --checkpoint /storage/sdawle/pi05_weights/pi05_libero_finetuned \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 \
  --num-episodes 1 \
  --steps-sweep 4 10 \
  --backend ttnn \
  --replan-steps 5 \
  --video-dir /storage/sdawle/libero_videos --video-fps 20
```

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | `pi05_libero_finetuned` | path to model.safetensors + normalizer stats |
| `--suite` / `--suites` | `libero_spatial` | one (`--suite`) or many (`--suites` nargs+) LIBERO suites |
| `--task-idx` / `--task-range` | `0` | single task or inclusive `(start, end)` range |
| `--num-episodes` | `3` | initial states per task (max 50; LIBERO ships 50 canonical inits per task) |
| `--max-steps` | per-suite default | env step cap; defaults: spatial=220, object=280, goal=300, libero_10=520 |
| `--backend` | `pytorch` | `pytorch` (CPU ref) or `ttnn` (Blackhole) |
| `--replan-steps` | `10` | apply this many actions per chunk before requesting a new chunk (openpi convention=5) |
| `--steps-sweep` | `10 4` | denoise-step counts to evaluate (one rollout per N) |
| `--video-dir` | none | write one mp4 per episode under `<dir>/N{N}/<suite>/task{XX}_ep{NN}_<title>_<success\|failure>.mp4` |
| `--video-fps` | `20` | playback fps (sim runs at 20 Hz) |

### Example: 5-episode sanity check

```bash
PYTHONPATH=$PWD:/storage/sdawle/libero_repo MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --num-episodes 5 --max-steps 220 --steps-sweep 4 \
  --backend ttnn --replan-steps 5
# → ~3 minutes wall, 5 episodes, prints per-episode success + final summary
```

### Example: full demo across all 4 suites with videos

```bash
PYTHONPATH=$PWD:/storage/sdawle/libero_repo MUJOCO_GL=osmesa HF_HOME=/storage/sdawle/hf_cache \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 --num-episodes 1 --steps-sweep 4 10 \
  --backend ttnn --replan-steps 5 \
  --video-dir /storage/sdawle/libero_videos
# → 80 episodes, ~50 min wall, 80 mp4s organized as N{4,10}/<suite>/...
```

### Latest measured task success (TTNN, 1 init/task, after the silu fix + perf port)

| Suite | N=4 | N=10 |
|---|---|---|
| libero_spatial | 9/10 (90%) | 9/10 (90%) |
| libero_object | 9/10 (90%) | 9/10 (90%) |
| libero_goal | 7/10 (70%) | 8/10 (80%) |
| libero_10 (long-horizon) | 7/10 (70%) | 7/10 (70%) |
| **total** | **32/40 (80%)** | **33/40 (82.5%)** |

Per-chunk inference: ~225 ms at N=4, ~490 ms at N=10 (untraced, includes per-chunk host→device transfers; trace mode is ~100 ms at N=10).

---

## Weights

The expert checkpoint must contain the adaRMS modulation tensors per layer:

```
model.layers.{i}.input_layernorm.dense.weight     # (3 * width, width)
model.layers.{i}.input_layernorm.dense.bias       # (3 * width,)             optional
model.layers.{i}.post_attention_layernorm.dense.weight
model.layers.{i}.post_attention_layernorm.dense.bias
```

…and the suffix checkpoint must contain `time_mlp_in.{weight,bias}` / `time_mlp_out.{weight,bias}` in addition to `action_in_proj` and `action_out_proj`. `state_proj` and `action_time_mlp_*` from PI0 are **not** used.

If your checkpoint uses different names, add a rename pass in `Pi0_5WeightLoader.state_dict` (already strips lerobot's `model.` prefix automatically).

---

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
