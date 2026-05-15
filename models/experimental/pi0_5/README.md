# PI0.5 (`pi0_5`) — Tenstorrent

End-to-end TTNN implementation of the **π₀.₅** (PI0.5) vision-language-action policy on Blackhole, with a PyTorch reference, LIBERO simulator integration, and a real-weights perf path that runs at **~64.85 ms / chunk** (~770 actions/s) with trace+2CQ.

This package is **self-contained** — it does not import from `models/experimental/pi0/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             PI0.5 Model                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────┐   ┌──────────────────────────┐│
│  │         PREFIX EMBEDDING            │   │    SUFFIX EMBEDDING      ││
│  │                                     │   │                          ││
│  │  ┌───────────┐   ┌───────────────┐  │   │           ┌────────┐     ││
│  │  │  Images   │   │ Lang + State  │  │   │           │ Noisy  │     ││
│  │  │  (224x224)│   │ (200 tokens,  │  │   │           │Actions │     ││
│  │  │  (3 views)│   │  state→bins)  │  │   │           │(50, 32)│     ││
│  │  └─────┬─────┘   └───────┬───────┘  │   │           └───┬────┘     ││
│  │        │                 │          │   │               │          ││
│  │        ▼                 │          │   │               ▼          ││
│  │  ┌───────────┐           │          │   │      ┌────────────────┐  ││
│  │  │  SigLIP   │           │          │   │      │ action_in_proj │  ││
│  │  │  Vision   │           │          │   │      │  (32 → 1024)   │  ││
│  │  │  Tower    │           │          │   │      └────────┬───────┘  ││
│  │  │(27 blocks)│           │          │   │               │          ││
│  │  └─────┬─────┘           │          │   │               │          ││
│  │        │                 │          │   │  ┌──────────┐ │          ││
│  │        ▼                 │          │   │  │ sincos(t)│ │          ││
│  │  ┌───────────┐           │          │   │  └─────┬────┘ │          ││
│  │  │Projector  │           │          │   │        ▼      │          ││
│  │  │(1152→2048)│           │          │   │  ┌──────────┐ │          ││
│  │  └─────┬─────┘           │          │   │  │ time_mlp │ │          ││
│  │        │                 │          │   │  │ in→silu→ │ │          ││
│  │        ▼                 ▼          │   │  │ out→silu │ │          ││
│  │  ┌───────────────────────────────┐  │   │  └─────┬────┘ │          ││
│  │  │  Image Embeds + Lang Embeds   │  │   │        │      │          ││
│  │  │  (Gemma 2B embedding)         │  │   │  adarms_cond  │          ││
│  │  └───────────────┬───────────────┘  │   └────────┼──────┼──────────┘│
│  │                  │                  │            │      │           │
│  └──────────────────┼──────────────────┘            │      │           │
│                     │                               │      │           │
│                     ▼                               │      ▼           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               DUAL-EXPERT TRANSFORMER (18 layers)                │  │
│  │  ┌────────────────────────┐    ┌────────────────────────┐        │  │
│  │  │     Gemma 2B VLM       │    │   Gemma 300M Expert    │        │  │
│  │  │   (processes prefix)   │    │ ★ AdaRMSNorm variant ★ │        │  │
│  │  │   Plain RMSNorm        │    │                        │        │  │
│  │  │                        │    │   adaRMS Dense per     │        │  │
│  │  │  Q_vlm ──┐             │    │   layer:               │        │  │
│  │  │  K_vlm ──┼─► SHARED ◄──┼────┼─ scale, shift, gate    │        │  │
│  │  │  V_vlm ──┘   ATTN      │    │   ← from adarms_cond   │        │  │
│  │  │                        │    │                        │        │  │
│  │  │  MLP_vlm               │    │   normed = RMS(x)      │        │  │
│  │  │                        │    │   out = normed*(1+s)+b │        │  │
│  │  │                        │    │   residual ← gate * .  │        │  │
│  │  └────────────────────────┘    └────────────────────────┘        │  │
│  └────────────────────────────────────┬─────────────────────────────┘  │
│                                       │                                │
│                                       ▼                                │
│                       ┌──────────────────────────────┐                 │
│                       │     FLOW MATCHING DENOISER   │                 │
│                       │     (10 denoising steps)     │                 │
│                       │                              │                 │
│                       │  for t in [1.0 → 0.0]:       │                 │
│                       │    v_t = action_out_proj(    │                 │
│                       │           expert_out)        │                 │
│                       │    x_t ← x_t + dt · v_t      │                 │
│                       └──────────────┬───────────────┘                 │
│                                      │                                 │
│                                      ▼                                 │
│                        ┌──────────────────────────┐                    │
│                        │      Action Output       │                    │
│                        │     [batch=1, 50, 32]    │                    │
│                        └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **Shared Attention**: VLM and Expert share K,V tensors (concatenated), but have separate Q and MLPs (same as PI0)
- **AdaRMSNorm in Expert**: each layer reads `(scale, shift, gate)` from a Dense projection of `adarms_cond` (time-derived). `out = RMSNorm(x)·(1+scale) + shift`; residual is gated.
- **State in language tokens**: 8-dim robot state → MEAN/STD normalize → discretize to 256 bins → append to prompt as `Task: …, State: <bins>; Action:`. No separate state token in the suffix.
- **Flow Matching**: same Euler integration as PI0; 10 (or 4) denoising steps from N(0,I) → actions.
- **Dual Experts**: VLM (2B) processes images+language+state-as-tokens; Expert (300M, adaRMS) processes only action tokens.

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
├── scripts/
│   └── download_weights.sh   # Download pi05_base / pi05_libero from HuggingFace
├── tests/
│   ├── pcc/                  # Reference-vs-spec correctness
│   └── perf/                 # Latency / throughput on Blackhole
└── weights/
    ├── pi05_base/            # lerobot/pi05_base safetensors (see download script)
    └── pi05_libero_finetuned/# lerobot/pi05_libero_finetuned_v044 (LIBERO eval)
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

The suite is layered: per-component → sub-model → end-to-end → diagnostic drilldowns. All TTNN-vs-torch tests need a Blackhole device and the real `pi05_base` weights.

```bash
# Run the whole suite
PYTHONPATH=$PWD python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/

# Component-level (TTNN module vs torch reference)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_gemma_vs_torch.py     # GemmaMLPTTNN
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_prefix_vs_torch.py    # PrefixEmbeddingTTNN
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_siglip_vs_torch.py    # SigLIPVisionTowerTTNN
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_suffix_vs_torch.py    # Pi0_5SuffixEmbeddingTTNN

# Sub-model
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_paligemma_vs_torch.py # PaliGemma backbone (VLM + Expert)

# End-to-end (full 10-step denoise)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model_vs_torch.py    # single seed
python models/experimental/pi0_5/tests/pcc/test_pcc_pi05_per_step_vs_torch.py                     # primary gate: 10-seed E2E sweep

# Diagnostic block-level drilldowns
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_vlm_block_drilldown.py     # one VLM block, sub-step PCC
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_expert_block_drilldown.py  # one AdaRMS expert block
```

**Primary gate:** `test_pcc_pi05_per_step_vs_torch.py` runs the real `sample_actions()` path across N seeds (default 10) and gates on the **mean** E2E PCC. E2E PCC is intrinsically seed-sensitive because flow-matching with random initial noise is a chaotic dynamical system — a 10-step Euler integration of a learned velocity field amplifies tiny per-step bf16 drift differently per input. The sweep is the implementation-quality signal; a single seed is not.

**Latest E2E PCC distribution (10-seed sweep, real `pi05_base` weights):**

| metric | value |
|---|---|
| mean | **0.991** |
| stdev | 0.007 |
| min | 0.978 |
| max | 0.998 |
| pass rate (PCC ≥ 0.95) | **10/10** |
| per-step velocity PCC (worst step, single seed) | 0.993 |

### Perf tests (Blackhole)

```bash
# Headline: full sample_actions with trace
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py
#  → Per-call avg ≈ 64-65 ms (N=10),  Action throughput ≈ 770 actions/s

# Without trace
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e.py
```

**Latest measured trace-mode perf (Blackhole, N=10, with SigLIP block-sharded encoder + adaRMS expert + sharded LayerNorm/RMSNorm):**

| metric | value |
|---|---|
| per-call latency | **64.85 ms** |
| chunk throughput | 15.42 chunks/s |
| action throughput | **770 actions/s** |
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

# 2. Download pi05_libero checkpoint (see "Weights" section above for details)
./models/experimental/pi0_5/scripts/download_weights.sh libero

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



---

## Weights

### Downloading from HuggingFace

Both checkpoints are hosted on HuggingFace. A convenience script handles the download:

```bash
# Install the HuggingFace CLI (if not already installed)
pip install huggingface_hub[cli]

# Log in (required — Gemma-family weights are gated)
huggingface-cli login

# Download both pi05_base and pi05_libero into weights/
./models/experimental/pi0_5/scripts/download_weights.sh

# Or download individually:
./models/experimental/pi0_5/scripts/download_weights.sh base    # lerobot/pi05_base (~14.5 GB)
./models/experimental/pi0_5/scripts/download_weights.sh libero  # lerobot/pi05_libero_finetuned_v044 (~7.5 GB)
```

| Checkpoint | HuggingFace repo | Size | Use |
|---|---|---|---|
| `pi05_base` | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | ~14.5 GB | PCC tests, perf tests, general inference |
| `pi05_libero_finetuned` | [`lerobot/pi05_libero_finetuned_v044`](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) | ~7.5 GB | LIBERO simulator evaluation |

---

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
