# PI0.5 (`pi0_5`) — Tenstorrent

End-to-end TTNN implementation of the **π₀.₅** (PI0.5) vision-language-action policy on Blackhole, with a PyTorch reference, LIBERO simulator integration, and a real-weights perf path that runs at **~65 ms / chunk** (~770 actions/s) with trace+2CQ.

This package is **self-contained** — it does not import from `models/experimental/pi0/`.

There is also an experimental **28-chip Blackhole Galaxy** spatial pipeline that
spreads the stages across a 32-chip mesh — see [BH Galaxy multi-chip pipeline](#bh-galaxy-multi-chip-pipeline-tttt_bh_glx).

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
│   ├── ttnn_pi0_5_model.py   # Pi0_5ModelTTNN (single-chip)
│   └── tt_bh_glx/            # experimental 28-chip BH Galaxy pipeline (see section below)
│       ├── pipeline.py       #   Pi0_5GLXPipeline end-to-end driver
│       ├── stages.py         #   chip layout (4 vision / 18 prefill / 6 denoise)
│       ├── mesh_setup.py     #   open 8x4 mesh + carve submeshes (FABRIC_2D)
│       ├── stage_{vision,prefill,denoise}.py + *_slice.py
│       ├── transport.py      #   fabric mesh-socket cross-chip transport
│       ├── kv_migration.py   #   layer-paired prefill→denoise KV migration
│       └── _l1_migration.py  #   denoise DRAM→L1 weight migration (PI0_GLX_DENOISE_L1)
├── eval/
│   └── libero_rollout.py     # LIBERO simulator → policy → success rate / videos
├── tests/
│   ├── pcc/                  # Reference-vs-spec correctness
│   └── perf/                 # Latency / throughput on Blackhole
└── weights/
    └── pi05_libero_upstream/ # Default checkpoint dir; safetensors + config.json + assets/
```

---

## Quickstart

### Inference (PyTorch reference)

```python
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

loader = Pi0_5WeightLoader("/path/to/pi05_libero_upstream")
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

```bash
# Source the validated production defaults (15 perf flags + 3-cam + checkpoint path).
source _bench_runs/pi05_production.env
```

```python
import ttnn
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=80_000_000)
ckpt   = "/path/to/pi05_libero_upstream"   # or $PI05_CHECKPOINT_DIR
loader = Pi0_5WeightLoader(ckpt)
# Use from_checkpoint(...) so action_horizon is auto-read from <ckpt>/config.json.
# Plain Pi0_5ModelConfig() defaults action_horizon=50, which silently wrong-sizes
# the denoise loop against the upstream pi05_libero checkpoint (trained at 10).
cfg    = Pi0_5ModelConfig.from_checkpoint(ckpt)
model  = Pi0_5ModelTTNN(cfg, loader, device)

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

## Performance tuning environment variables

All pi0.5 perf flags + the canonical config live in **`_bench_runs/pi05_production.env`** — a single source of truth that all perf tests, the LIBERO 400-ep orchestrator, and any manual `pytest` invocation should source first.

```bash
# Source once per shell. Defaults applied = validated 97.2% LIBERO config (commit e52d0d23cff).
source _bench_runs/pi05_production.env

# Then run anything:
pytest models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py -s
bash   _bench_runs/libero_400ep_sweep.sh

# Override individual flags AFTER sourcing:
source _bench_runs/pi05_production.env
PI0_NUM_CAMERAS=2 PI05_NUM_DENOISE_STEPS=5 pytest …
```

### Full flag table

| Flag | Default (env file) | What it does |
|---|---|---|
| **Matmul / kernel tuning** | | |
| `PI0_EXPERT_MM_LOFI` | `1` | Denoise expert matmuls at LoFi compute fidelity (PCC-safe). |
| `PI0_ROPE_TABLES_L1` | `1` | Resident RoPE sin/cos tables in L1. |
| `PI0_MM_SWEEP_V2` | `1` | Enable tuned matmul picker v2 (BH-recalibrated). |
| `PI0_DENOISE_MM_TUNE` | `1` | Per-shape denoise matmul overrides. |
| `PI0_PREFILL_MM_TUNE` | `1` | Per-shape VLM prefill matmul overrides. |
| **Attention / mask handling** | | |
| `PI0_UPSTREAM_MASKS` | `1` | Upstream-compat attention mask + RoPE tables (pre-staged before trace capture). |
| `QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT` | `1` | Per-head parallelism in `NlpConcatHeads`. |
| `QWEN_NLP_CREATE_HEADS_HEAD_SPLIT` | `1` | Per-head parallelism in `NlpCreateQkvHeads`. |
| `PI0_MQA_HEAD_SPLIT` | `1` | C++ MQA-aware Q-head parallelism — splits the 1-core denoise NlpCreateHeads into 8 cores (−1.42 ms / −67% on that op). |
| `PI0_SDPA_DENOISE_K_FORCE` | `96` | SDPA denoise k_chunk override (1056 = 96×11 exact divisor — saves ~0.5 ms / inference vs k=64). |
| **VLM single-pass at bs=3** | | |
| `PI0_NUM_CAMERAS` | `3` | Image slots fed to SigLIP. **3 matches openpi training spec** (LIBERO sim has 2 real cameras + 1 pad slot). Set to `2` for the fast path (saves ~14 ms but skips the training-required pad — see "Honest accounting" below). |
| `PI0_VLM_CHUNK_SIZE` | `1024` | Single-pass VLM prefill chunk (3·256 image + 256 lang = 1024). At `NUM_CAMERAS=2` use `768`. |
| `PI0_VLM_MLP_BF8_OUT` | `1` | VLM gate/up/down outputs in `bfloat8_b` (validated ≥0.996 PCC). |
| `PI0_VLM_MLP_MINIMAL` | `1` | VLM gate/up routed through `ttnn.experimental.minimal_matmul`. |
| `PI0_VLM_MINIMAL_CFG` | `4,8,8,1,8` | `minimal_matmul` tile config: `M,K,N,subblock_h,subblock_w`. |
| **Checkpoint** | | |
| `PI05_CHECKPOINT_DIR` | `/home/tt-admin/pi05_cache/pi05_libero_upstream` | Checkpoint directory (machine-specific; override in your shell rc if path differs). |
| **NOT in env file — set explicitly per run** | | |
| `PI05_NUM_DENOISE_STEPS` | *(unset → 10)* | Diffusion solver step count. **10 is the openpi training default.** Set to `5` for the perf-tuned path (~45 ms inference, validated 97.2% LIBERO, halves device time at <0.4% accuracy cost). |
| `PI0_TOKENIZER_PATH` | *(no default)* | PaliGemma tokenizer path; needed only by `eval/libero_rollout.py`. |
| `LIBERO_REPO_PATH` | *(no default)* | LIBERO repo clone; needed only by `eval/libero_rollout.py`. |
| `MUJOCO_GL` | *(unset)* | `osmesa` for headless render under LIBERO. |

### Honest accounting — what's a real default vs a perf shortcut

| Knob | Training-spec value | Our env file default | Notes |
|---|---|---|---|
| `action_horizon` | from `<ckpt>/config.json` (10 for pi05_libero) | auto-read via `Pi0_5ModelConfig.from_checkpoint(...)` | The classmethod is the only safe constructor; bare `Pi0_5ModelConfig()` silently uses 50. |
| `num_cameras` | 3 (per `Pi0Config.inputs_spec`) | **3** ✓ | LIBERO sim only has 2 real cameras → adapter pads slot 3 with `-1.0`. Using 2 skips the pad ⇒ -14 ms but deviates from training input shape. |
| `num_denoising_steps` | 10 | **NOT defaulted** (set explicitly) | 5 steps still reaches 97% LIBERO but is an inference-time tradeoff, not a stealth default. |
| `chunk_size` | derived | 1024 (matches `NUM_CAMERAS=3`) | Must change to 768 if `NUM_CAMERAS=2`. |

---

## Tests

All tests below are skipped automatically if the default checkpoint (`models/experimental/pi0_5/weights/pi05_libero_upstream/model.safetensors`, or whatever `$PI05_CHECKPOINT_DIR` points at) is missing.

### PCC (correctness) tests

```bash
# Run all PCC tests (no device required for some; the *_real_weights tests
# need a Blackhole device for the TTNN path)
PYTHONPATH=$PWD python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/

# Individual:
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_suffix.py            # suffix layer (sincos+MLP, time_mlp_out silu)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_adarms_gemma.py      # AdaRMS Gemma block
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_e2e_reference.py     # E2E reference (no device)
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_real_weights.py      # E2E pytorch with real pi05_libero_upstream weights
python_env/bin/pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_ttnn_real_weights.py # E2E TTNN with real weights — needs device

# Per-step velocity + 10-seed e2e PCC distribution (the headline E2E correctness number):
PYTHONPATH=$PWD python models/experimental/pi0_5/tests/pcc/test_pcc_pi05_per_step_vs_torch.py
# Custom seed list:
PI0_PCC_SEEDS="42,7,100" python models/experimental/pi0_5/tests/pcc/test_pcc_pi05_per_step_vs_torch.py
```

### E2E PCC: why we report a distribution, not a single seed

The e2e PCC of a 10-step flow-matching Euler integrator is intrinsically **seed-sensitive**: each rollout is a 10-step nonlinear ODE solve starting from random initial noise, so per-step bf16 drift is amplified differently for every input pattern. Single-seed PCC has stdev ≈ 0.006 (≈ 2% peak-to-peak range across seeds) — it is a Monte Carlo sample, not a precision metric.

The right thing to report is the **mean e2e PCC across N seeds**. `test_pcc_pi05_per_step_vs_torch.py` runs a 10-seed sweep by default and gates on `mean ≥ 0.95`.

**Latest measured E2E PCC distribution (10 seeds, Blackhole, pi05_libero_upstream weights, SigLIP BS on):**

| metric | value |
|---|---|
| mean    | **0.9910** |
| median  | 0.9929 |
| stdev   | 0.0070 |
| min     | 0.9781 |
| max     | 0.9980 |
| ≥ 0.99  | 6 / 10 seeds |
| ≥ 0.95  | 10 / 10 seeds |

Per-step velocity PCC (worst across 10 denoise steps, single representative seed): **0.9933** (cosine ≈ 0.9994 per step; vs TTNN-internal bf16 trajectory).

### Perf tests (Blackhole)

```bash
# Headline: full sample_actions with trace + 2CQ (the published number)
PYTHONPATH=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py
#  → Per-call avg ≈ 64-65 ms (N=10),  Action throughput ≈ 770 actions/s

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

**Latest measured trace-mode perf (Blackhole, N=10, with ViT-BH-style block-sharded SigLIP encoder data path):**

| metric | value |
|---|---|
| per-call latency | **64.85 ms** |
| chunk throughput | 15.42 chunks/s |
| action throughput | **770.98 actions/s** |
| jitter (stddev) | 0.06 ms |
| trace capture (one-time) | ~410 ms |

The SigLIP encoder runs entirely in **L1 block-sharded layout on a common 12×8 = 96-core grid** (ViT-BH tech report §5.3 pattern). Hidden states stay block-sharded across all 27 encoder layers — only re-tiling for SDPA, which uses the full 13×10 grid. The 12×8 grid is the largest divisor-clean choice given hidden=1152 (36 tiles, divisible by 12) and M=512 (16 tiles, divisible by 8); going wider (e.g. 13×10) would need >8% weight/compute padding to keep tile divisibility, which doesn't pay back. Runtime master switch: `PI0_SIGLIP_BS=0` reverts to the interleaved-LN baseline at ~65.1 ms with no rebuild.

---

## BH Galaxy multi-chip pipeline (`tt/tt_bh_glx/`)

Everything above runs the model on a **single Blackhole chip**. `tt/tt_bh_glx/`
is a separate, experimental **28-chip spatial pipeline** that spreads the three
stages across a Blackhole **Galaxy** (8×4 = 32-chip parent mesh) and streams
activations chip-to-chip over fabric sockets. It reuses the same TTNN building
blocks (`ttnn_siglip`, `ttnn_gemma`, `ttnn_suffix`) but composes them as
per-chip slices instead of one device's full model.

> **Status (experimental).** Two paths exist:
> - **Eager socket pipeline** (`pipeline.py`; FABRIC_2D, per-chip 1×1 submeshes) —
>   functional end-to-end (PCC 0.9957 vs torch). Its parent-rooted trace capture
>   (`pipeline.py::capture_trace`) does **not** work — it records an empty trace and
>   deadlocks in `end_trace_capture` (ops live on the 1×1 children, not the parent).
> - **Fully-traced per-stage pipeline** (`tests/perf/_trace_e2e_full.py`; FABRIC_1D,
>   per-stage single-mesh + `point_to_point`, on-device KV migration) — all three
>   stages traced, **PCC 0.9988 vs torch, ~82 ms/inference**. This is the working
>   traced path. See **`tt/tt_bh_glx/TRACED_PIPELINE_JOURNEY.md`** (design + diagrams)
>   and **`tests/perf/TRACED_E2E_PERF.md`** (numbers), and the "Traced pipeline"
>   subsection below.

### Chip layout (`stages.py`)

```
   col→  0 1 2 3
row↓  0  V V V V    V = vision   4 chips   shape (1,4) offset (0,0)
      1  P P P D    P = prefill  18 chips  shape (6,3) offset (1,0)
      2  P P P D    D = denoise  6 chips   shape (6,1) offset (1,3)
      3  P P P D    (row 7 = 4 spare chips)
      …  P P P D
      6  P P P D
      7  . . . .
```

| Stage | Chips | Contents |
|---|---|---|
| **vision** | 4 | chip0 = patch_embed + pos_emb; chips1–3 = SigLIP layers 0–8 / 9–17 / 18–26 + post_ln + mm_projector |
| **prefill** | 18 | 1 Gemma-2B VLM block per chip; each chip keeps its layer's KV |
| **denoise** | 6 | 3 AdaRMS Gemma-300M expert blocks per chip + replicated suffix MLP; runs the N-step Euler loop |

**Per-call flow** (`Pi0_5GLXPipeline.sample_actions`): vision → socket → build
prefix (reshape image embeds + embed lang tokens) → 18-chip VLM prefill →
layer-paired KV migration (prefill chip `i` → denoise chip `i // 3`) → N-step
denoise loop with on-device fp32 Euler integration → slice to `action_horizon`.

**Transport.** Cross-chip handoff uses fabric **mesh sockets**
(`ttnn.experimental.send_direct_async` / `recv_direct_async`, `FABRIC_2D`) — the
sender writes directly into the receiver's pre-allocated buffer, no host bounce.
Set `PI05_GLX_TRANSPORT=host` to fall back to the legacy host-bounce path for
A/B testing without a rebuild.

### Quickstart

```python
from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.tt_bh_glx.pipeline import Pi0_5GLXPipeline

ckpt   = "/path/to/pi05_libero_upstream"
cfg    = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(ckpt), num_denoising_steps=5)
loader = Pi0_5WeightLoader(ckpt)

# open_galaxy_mesh carves the 8x4 parent into the 4/18/6 submeshes (FABRIC_2D),
# and closes them parent-last on exit.
with open_galaxy_mesh(l1_small_size=24576) as h:
    pipeline = Pi0_5GLXPipeline(cfg, loader.categorized_weights, h)
    actions, timings = pipeline.sample_actions(
        images=[img0, img1, img2],                # list of (1, 3, 224, 224) torch tensors
        img_masks=[m0, m1, m2],
        lang_tokens=tokens,                       # (1, lang_len) int64
        lang_masks=lang_masks,
    )
    # actions: (1, action_horizon, action_dim); timings: per-stage StageTimings (ms)
```

### Performance flags (in addition to the single-chip flags above)

| Flag | Default | What it does |
|---|---|---|
| `PI0_GLX_DENOISE_L1` | `1` (ON) | Migrate the static denoise-stage weights/biases + per-call prefix KV from DRAM into L1 so the N-step Euler loop reads them on-chip instead of re-streaming ~93 MB/chip from DRAM each step. Set `0` to revert to DRAM. |
| `PI05_GLX_TRANSPORT` | *(socket)* | `host` falls back to the legacy host-bounce transport. |
| `PI0_ROPE_TABLES_L1` | *(off here)* | Keep upstream-compat RoPE tables in L1. |
| `PI05_NUM_DENOISE_STEPS` | `5` (in the GLX tests) | Euler step count. |
| `PI0_NUM_CAMERAS` | `3` | Image slots fed to vision (3 = training spec). |
| `PI05_GLX_NUM_WARMUP` / `PI05_GLX_NUM_ITERS` | `1` / `3` (e2e), `5` (stages) | Perf-test warmup + timed-iteration counts. |

**Denoise L1 residency.** The 3 expert layers/chip (~93 MB: bf8 matmuls +
bf16 adaRMS mod) fit inside the ~180 MB usable L1 per Blackhole chip. Denoise is
the easy case — the expert MLP (4096) is 4× smaller than the VLM MLP (16384), so
the interleaved-L1 weights don't clash with the matmul kernel's static CB region.
The migration runs once at `Pi0_5GLXPipeline.__init__` (see `_l1_migration.py`)
and only touches the denoise stage; prefill/vision placement is unchanged.

### Tests (require a 32-chip BH Galaxy + `PI05_CHECKPOINT_DIR`)

```bash
source _bench_runs/pi05_production.env   # checkpoint path + production knobs

# Mesh-carve smoke + per-stage PCC (vision ≥0.997, prefill ≥0.99, denoise ≥0.95):
PYTHONPATH=$PWD TT_METAL_HOME=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py

# End-to-end PCC vs torch Pi0_5Model (target ≥0.95):
PYTHONPATH=$PWD TT_METAL_HOME=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_e2e.py

# Per-stage wall-clock + host-bounce microbench:
PYTHONPATH=$PWD TT_METAL_HOME=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_stages.py

# End-to-end wall-clock + per-stage StageTimings breakdown:
PYTHONPATH=$PWD TT_METAL_HOME=$PWD python_env/bin/pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_e2e.py
```

### Latest measured (eager, 5 denoise steps, 3 cameras, sockets)

| metric | value |
|---|---|
| E2E PCC vs torch | **0.9957** |
| per-stage PCC | vision 0.9997 · prefill 0.997 · denoise-chain 0.961 |
| per-call latency | **~276 ms** (3.62 chunks/s) |
| stage breakdown (ms) | vision 37.1 · v→p 0.1 · prefill 31.4 · kv_mig 1.6 · **denoise 202.1** |

Denoise dominates wall-clock; the `PI0_GLX_DENOISE_L1` path above targets it.
The ~276 ms baseline was measured **before** the L1 change landed — re-run
`test_perf_tt_bh_glx_e2e.py` with `PI0_GLX_DENOISE_L1=1` vs `=0` to measure the
delta on your machine.

---

## Traced pipeline (`tt/tt_bh_glx/` — the working trace path)

The fully-traced pipeline captures **all three stages** and matches torch at
**PCC 0.9988** at **~82 ms / inference** (5 denoise steps). Full design rationale +
ASCII diagrams: **`tt/tt_bh_glx/TRACED_PIPELINE_JOURNEY.md`**; perf numbers:
**`tests/perf/TRACED_E2E_PERF.md`**.

### Why it differs from the eager socket pipeline

- A TTNN mesh trace only records ops on **that mesh's** command queue. The eager
  pipeline issues ops on per-chip **1×1 submeshes** but captures on the **parent** →
  empty trace + full-mesh-finish **deadlock**. (Capturing *per-submesh* fixes this —
  validated by `tests/perf/_socket_trace_2x2.py`.)
- Fix: run **each stage as one single-mesh SPMD computation** with in-mesh
  `point_to_point` hand-offs, captured as **one trace per stage**, under **FABRIC_1D**
  (`point_to_point` traces+replays under 1D; hangs under 2D).
- Sockets are kept only for the **eager, one-shot cross-stage hand-offs**
  (`send_direct_async`); they're not used inside a trace.

### Traced chip layout (collinear, distinct from the eager layout above)

```
          col 0      col 1      col 2      col 3
 row 0..5  prefill    prefill    prefill    denoise     prefill (6,3)@(0,0) SNAKE; denoise (6,1)@(0,3)
 row 6     vision     vision     vision     embed       vision (1,3)@(6,0); embed (1,1)@(6,3)
 row 7     scratch     --         --         --
```

Chips are placed so every cross-stage hand-off is **adjacent + collinear** (FABRIC_1D
sockets are adjacent-only): prefill uses a boustrophedon **snake** so consecutive
layers stay collinear; vision tail `(6,2)`→prefill `(5,2)`; prefill row `r` `(r,2)`→
denoise `(r,3)`.

### Fabric optimizations (committed)

- **Multi-core/multi-link `point_to_point`** (C++): 2.7 → 5.3 GB/s (2 cores/2 links;
  trace-safe). Used inside the traced stages.
- **Multi-connection sockets** (`PI05_SOCK_CONN=2`): up to 15.5 GB/s. Used for the
  eager cross-stage hand-offs.
- **On-device KV migration** (p2p-gather + adjacent socket): ~265 ms host-bounce →
  ~11 ms.

> Build note: the C++ `point_to_point` change is in source, but `cmake --build build`
> writes to `build_Release/ttnn/` while Python loads `ttnn/ttnn/_ttnn.so` +
> `build_Release/lib/_ttnncpp.so`. After building, sync them:
> `cp -f build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so && cp -f build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so`.

### How to run — per-stage trace validations

These are standalone scripts (not pytest). All need a 32-chip BH Galaxy +
`PI05_CHECKPOINT_DIR`; the full 8×4 mesh must be opened (a bare 2×2 can't train
fabric on the Galaxy). Reset between runs with `tt-smi -glx_reset`.

```bash
export PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
P=python_env/bin/python

# --- per-stage traced compute (each captures one trace, validates vs torch) ---
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_trace_e2e_vision.py        # vision: PCC ~0.9997
tt-smi -glx_reset; REPRO_PREFIX=1024 $P models/experimental/pi0_5/tests/perf/_trace_prefill_stage_repro.py  # prefill snake (source the prod env first for the matmul tuning)
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_trace_denoise_stage_repro.py   # denoise Euler loop
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_trace_siglip_stage_repro.py    # SigLIP block chain

# --- fabric mechanism repros ---
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_socket_multimesh_repro.py   # multi-chip-mesh collinear socket
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_kv_socket_repro.py          # KV p2p-gather + adjacent socket
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_p2p_microbench.py           # point_to_point bandwidth sweep
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_p2p_multicore_probe.py      # link count + multi-core socket BW
tt-smi -glx_reset; $P models/experimental/pi0_5/tests/perf/_socket_trace_2x2.py         # socket-in-1x1-trace replay (2-chip)
```

### How to run — full traced e2e (`_trace_e2e_full.py`)

Production perf flags are **baked in** via `os.environ.setdefault` (no need to source
the env). Modes are selected by env var:

```bash
export PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
P=python_env/bin/python; T=models/experimental/pi0_5/tests/perf/_trace_e2e_full.py

tt-smi -glx_reset; $P $T                       # run once: finite actions, shapes
tt-smi -glx_reset; PI05_E2E_PCC=1   $P $T       # + torch-reference PCC (target ≥0.95; gets ~0.9988)
tt-smi -glx_reset; PI05_E2E_PERF=1  $P $T       # + 2-warmup/20-iter traced-replay latency table
tt-smi -glx_reset; PI05_E2E_TIMING=1 $P $T      # + host-bounce vs on-device KV migration timing
tt-smi -glx_reset; PI05_PREFILL_PROFILE=1 $P $T # + prefill snake block/p2p/layout split
tt-smi -glx_reset; PI05_VISION_SOCKET=1 PI05_E2E_PCC=1 $P $T   # fully on-device vision->prefix (else host-bounce)
```

| Env var | Default | Effect |
|---|---|---|
| `PI05_E2E_PCC` | off | run torch `Pi0_5Model.sample_actions` reference + report PCC |
| `PI05_E2E_PERF` | off | 2-warmup + 20-iter traced-replay latency (override `PI05_E2E_PERF_WARMUP`/`_ITERS`) |
| `PI05_E2E_TIMING` | off | host-bounce vs on-device KV-migration timing |
| `PI05_PREFILL_PROFILE` | off | eager prefill block-fwd / layout-conv / p2p split |
| `PI05_VISION_SOCKET` | off (host) | build prefix on-device + socket vision→prefill (needs ~1 GB embed table on vision mesh) |
| `PI05_SOCK_CONN` | `2` | socket connections (= worker cores) per cross-stage hand-off |

### How to run — PCC & perf pytest (eager pipeline)

The pytest suite below targets the **eager** socket pipeline (the traced path is the
standalone scripts above):

```bash
source _bench_runs/pi05_production.env   # checkpoint + production knobs
PP="PYTHONPATH=$PWD TT_METAL_HOME=$PWD python_env/bin/pytest -xvs"

$PP models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py   # per-stage PCC
$PP models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_e2e.py      # e2e PCC vs torch (≥0.95)
$PP models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_stages.py # per-stage wall-clock
$PP models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_e2e.py    # e2e wall-clock + StageTimings
```

### Latest measured (traced, 5 denoise steps, 3 cameras)

| metric | value |
|---|---|
| E2E PCC vs torch | **0.9988** |
| per-inference latency (warm) | **~82 ms** |
| traced replays | vision 8.0 · prefill 32.5 · denoise 26.1 ms |
| KV migration | ~265 ms host-bounce → **~11 ms** on-device |

---

## All-socket per-1×1-mesh traced e2e (`tt/tt_bh_glx/socket_trace_experiment/`)

A second, fully-resolved traced path that — unlike the per-stage `_trace_e2e_full.py`
above — keeps **every chip a 1×1 submesh** and makes **every** cross-stage hand-off a
**fabric socket** (`send_direct_async`/`recv_direct_async`, **no** `point_to_point`),
then **traces per-submesh** (begin/end_trace_capture on each of the 28 submeshes, not the
parent). It reuses the production pipeline's pure-device body (`_sample_actions_device`)
unchanged and does **not** modify the production pipeline.

**Status: RESOLVED — it works.** Captures per-submesh (28 concurrent submesh traces, no
deadlock) and replays at **PCC 1.000000 vs eager / 0.998796 vs torch**, including the
N-step denoise loop. The original `capture_trace` hang was capturing on the PARENT while
ops ran on 1×1 children — not a socket-replay problem. Full writeup:
**`tt/tt_bh_glx/socket_trace_experiment/README.md`**.

### How to run — full traced socket e2e (`run_socket_traced.py`)

Production perf flags are **auto-applied** from `_bench_runs/pi05_production.env` (via
`os.environ.setdefault` at startup) — no manual `source` needed. Only the checkpoint path
and `PYTHONPATH`/`TT_METAL_HOME` are machine-specific. **Always start from a clean
`tt-smi -glx_reset`** — without the production flags or on a dirty device it runs ~67 ms
at PCC ~0.98 instead of ~50 ms at PCC 1.0.

```bash
cd /home/tt-admin/sdawle/tt-metal

export PI05_CHECKPOINT_DIR=/home/tt-admin/sdawle/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD

python_env/bin/tt-smi -glx_reset                 # clean device state — do this before EVERY run

# capture + PCC-vs-eager + PCC-vs-torch + PERF_ITERS-averaged latency
TRACE_SCOPE=full FIXED_NOISE=1 PI05_E2E_PCC=1 PI05_SOCK_CONN=2 PERF_ITERS=20 \
  python_env/bin/python models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/run_socket_traced.py
```

Bare `python_env/bin/python .../run_socket_traced.py` (after the reset) also works — every
flag below defaults correctly; the line above just makes them explicit and adds the
torch-PCC check.

| Env var | Default | Effect |
|---|---|---|
| `TRACE_SCOPE` | `full` | whole pipeline (vision → build_prefix → prefill → KV migration → N-step denoise). Also `vp` (vision+prefill) and `denoise` (denoise-only) for localization. |
| `FIXED_NOISE` | `1` | pin one noise tensor across eager+capture+replay. **Required** for a valid PCC compare (else fresh `torch.randn` per call makes eager≠replay). |
| `PI05_E2E_PCC` | off | after replay, also compare all-socket actions vs the torch reference (target ≥0.95; gets ≈0.998796). |
| `PI05_SOCK_CONN` | `2` | socket connections per hop. `2` spreads `send_direct_async` across the adjacent pair's 2 fabric links (~7% faster e2e); `1` is the single-link baseline. |
| `PERF_ITERS` | `20` | replays averaged for the `PERF:` latency line. |
| `EAGER` | off | **no trace** — 1 warm-up + 1 profiled eager iter with tracy signposts. Use under `python -m tracy` for true per-op device-kernel durations. |
| `TRACY` | off | capture + 1 warm-up + 1 profiled replay, then stop. Use under `python -m tracy --device-trace-profiler`. |

Prints, per replay, `PCC vs eager` (trace fidelity, expect 1.000000), then `PCC vs torch`
(numerics, ≈0.998796), then `PERF: traced all-socket e2e replay = <ms>/inference`.

### Latest measured (validated 2026-06-14, clean reset, production flags on)

| sockets | e2e replay | infer/s | PCC vs eager | PCC vs torch |
|---|---|---|---|---|
| `PI05_SOCK_CONN=1` | 53.67 ms | 18.6 | 1.000000 | 0.998796 |
| `PI05_SOCK_CONN=2` *(default)* | **49.94 ms** | **20.0** | 1.000000 | 0.998796 |

2 connections is ~7% faster at identical numerics. The gain is modest (not 2×) because the
pipeline is **serialization/dispatch-bound, not socket-bandwidth-bound** — the busiest
single chip does only ~4 ms of compute; the ~50 ms is the critical path through the 18-hop
prefill snake + the N-step denoise loop + host dispatch.

---

## LIBERO simulator rollout

End-to-end real-robot benchmark on the LIBERO suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`).

### One-time setup (not in git)

These artifacts are not tracked. Pick paths under your own `$HOME` — the
old hardcoded `/storage/sdawle/...` paths are gone on most machines.

```bash
# 0. Working dir convention (used in commands below).
export PI05_BASE=$HOME/pi05_cache         # any writable dir
mkdir -p $PI05_BASE/tokenizer $PI05_BASE/weights

# 1. PaliGemma tokenizer (~4 MB; public GCS, no auth)
curl -L -o $PI05_BASE/tokenizer/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# 2. pi05 checkpoint — pick ONE:
#    (a) lerobot/pi05_base finetuned weights (recommended for accuracy testing):
#        https://huggingface.co/lerobot/pi05_libero_finetuned    [public]
#    (b) openpi pi05_libero_upstream (recommended for upstream parity):
#        https://huggingface.co/openpi/pi05_libero               [gated; needs HF auth]
#
#    Expected layout under <ckpt_dir>:
#      model.safetensors                                      ~7.2 GB, bf16 weights
#      config.json                                            5-key header (action_dim, action_horizon, paligemma_variant, action_expert_variant, precision)
#      assets/physical-intelligence/libero/norm_stats.json    state/action mean/std/q01/q99
#
#    NOTE on config.json provenance:
#    The canonical Orbax checkpoint at gs://openpi-assets/checkpoints/pi05_libero/
#    is JAX format (16 files, ~7.2 GB across .ocdbt chunks) and has NO config.json.
#    The safetensors mirror on HuggingFace was produced by openpi/lerobot conversion
#    scripts that ALSO author the minimal config.json. If you bypass HF and convert
#    the Orbax checkpoint yourself, you must hand-write config.json with at minimum:
#        {"action_dim": 32, "action_horizon": 10, "paligemma_variant": "gemma_2b",
#         "action_expert_variant": "gemma_300m", "precision": "bfloat16"}
#    Without it, Pi0_5ModelConfig.from_checkpoint(...) falls back to action_horizon=50,
#    which over-sizes the denoise loop against the upstream pi05_libero checkpoint
#    (trained at 10). Symptom: +2 ms wall-clock, PCC degraded ~0.027.
#
#    The norm_stats.json is fetchable without auth directly from the GCS bucket:
#      curl -L -o $PI05_BASE/weights/pi05_libero_upstream/assets/physical-intelligence/libero/norm_stats.json \
#        https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json

# 3. LIBERO env from source (the PyPI package is broken)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $PI05_BASE/libero_repo

# 4. System packages for MuJoCo headless render (one-time, needs sudo)
sudo apt install -y libosmesa6 libegl1-mesa xvfb ffmpeg

# 5. Python deps — install LIBERO's runtime deps INTO python_env.
#    This venv is uv-managed; use uv pip (not pip). Both tt-metal-cache and
#    uv-cache must be redirected because $HOME/.cache may be a dangling symlink
#    on shared hosts (see [[reference_libero_config]] memory).
#    DO NOT pass 'numpy<2' or similar version pins — they would downgrade
#    tt-metal's numpy 1.26 and break ttnn imports.
#    robosuite 1.4.0 is REQUIRED (newer 1.5.x moved single_arm_env and breaks
#    libero 0.1.0's import path).
export UV_CACHE_DIR=$PWD/.tt_metal_cache/uv && mkdir -p $UV_CACHE_DIR
export VIRTUAL_ENV=$PWD/python_env
uv pip install "robosuite==1.4.0" mujoco bddl easydict cloudpickle gym imageio-ffmpeg

# 6. Install the libero package itself (--no-deps to avoid old pin downgrades).
#    This is an editable install; the package import path uses
#    $PI05_BASE/libero_repo as a namespace-package root.
uv pip install --no-deps -e $PI05_BASE/libero_repo
```

Verify the install:
```bash
python_env/bin/python -c "
import ttnn, torch, numpy, robosuite, mujoco
from libero.libero.envs import OffScreenRenderEnv
print('ttnn:', 'ok'); print('numpy:', numpy.__version__); print('torch:', torch.__version__)
print('robosuite:', robosuite.__version__); print('mujoco:', mujoco.__version__)
print('libero env import: ok')
" 2>&1 | tail -8
# Expected: numpy 1.26.x, torch ≥2.7, robosuite ≥1.5, mujoco ≥3.x, libero ok.
```

### Running a rollout

```bash
cd /home/tt-admin/sdawle/pi0/tt-metal

# All 15 perf flags + PI05_CHECKPOINT_DIR come from one source of truth.
source _bench_runs/pi05_production.env

PI0_TOKENIZER_PATH=$PI05_BASE/tokenizer/paligemma_tokenizer.model \
LIBERO_REPO_PATH=$PI05_BASE/libero_repo \
MUJOCO_GL=osmesa \
TT_METAL_CACHE=$PWD/.tt_metal_cache \
TT_METAL_HOME=$PWD \
PYTHONPATH=$PWD:$PI05_BASE/libero_repo \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --checkpoint $PI05_CHECKPOINT_DIR \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 \
  --num-episodes 1 \
  --steps-sweep 10 \
  --backend ttnn \
  --action-horizon 10 \
  --state-in-prompt false \
  --replan-steps 5 \
  --video-dir $PI05_BASE/libero_videos --video-fps 20
```

Required env vars NOT in `pi05_production.env` (machine/task-specific):
- `PI0_TOKENIZER_PATH` — PaliGemma SentencePiece model file.
- `LIBERO_REPO_PATH` — LIBERO repo clone (the BDDL task descriptions live here).
- `TT_METAL_CACHE` — required if `$HOME/.cache` symlinks to a missing target.
- `MUJOCO_GL=osmesa` — headless render backend.

CLI flag notes:
- `--state-in-prompt false --action-horizon 10` — correct for the **upstream
  openpi pi05_libero** checkpoint. Use `true / 50` for the lerobot finetune.
- `--steps-sweep 10` runs at the openpi training default (10 denoise steps).
  Use `--steps-sweep 5` for the perf-tuned path (~half the device time).

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | `/storage/sdawle/pi05_weights/pi05_libero_upstream` | path to model.safetensors + config + assets/. Stale default — pass yours. |
| `--suite` / `--suites` | `libero_spatial` | one (`--suite`) or many (`--suites` nargs+) LIBERO suites |
| `--task-idx` / `--task-range` | `0` | single task or inclusive `(start, end)` range |
| `--num-episodes` | `3` | initial states per task (max 50; LIBERO ships 50 canonical inits per task) |
| `--max-steps` | per-suite default | env step cap; defaults: spatial=220, object=280, goal=300, libero_10=520 |
| `--backend` | `pytorch` | `pytorch` (CPU ref) or `ttnn` (Blackhole) |
| `--replan-steps` | `10` | apply this many actions per chunk before requesting a new chunk (openpi convention=5) |
| `--steps-sweep` | `10 4` | denoise-step counts to evaluate (one rollout per N) |
| `--action-horizon` | `10` | chunk size; use 10 for upstream openpi pi05_libero, 50 for lerobot finetune |
| `--state-in-prompt` | `false` | embed robot state as discretized bins in the prompt; use `true` for lerobot finetune |
| `--video-dir` | none | write one mp4 per episode under `<dir>/N{N}/<suite>/task{XX}_ep{NN}_<title>_<success\|failure>.mp4` |
| `--video-fps` | `20` | playback fps (sim runs at 20 Hz) |

### Example: 5-episode sanity check

```bash
PI0_TOKENIZER_PATH=$PI05_BASE/tokenizer/paligemma_tokenizer.model \
LIBERO_REPO_PATH=$PI05_BASE/libero_repo \
MUJOCO_GL=osmesa TT_METAL_CACHE=$PWD/.tt_metal_cache TT_METAL_HOME=$PWD \
PYTHONPATH=$PWD:$PI05_BASE/libero_repo \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --checkpoint $PI05_BASE/weights/pi05_libero_upstream \
  --num-episodes 5 --max-steps 220 --steps-sweep 10 \
  --backend ttnn --replan-steps 5 \
  --action-horizon 10 --state-in-prompt false
# → ~3 minutes wall, 5 episodes, prints per-episode success + final summary
```

### Example: full demo across all 4 suites with videos

```bash
PI0_TOKENIZER_PATH=$PI05_BASE/tokenizer/paligemma_tokenizer.model \
LIBERO_REPO_PATH=$PI05_BASE/libero_repo \
MUJOCO_GL=osmesa TT_METAL_CACHE=$PWD/.tt_metal_cache TT_METAL_HOME=$PWD \
PYTHONPATH=$PWD:$PI05_BASE/libero_repo \
python_env/bin/python -u models/experimental/pi0_5/eval/libero_rollout.py \
  --checkpoint $PI05_BASE/weights/pi05_libero_upstream \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 --num-episodes 1 --steps-sweep 10 \
  --backend ttnn --replan-steps 5 \
  --action-horizon 10 --state-in-prompt false \
  --video-dir $PI05_BASE/libero_videos
# → 40 episodes, ~25 min wall, 40 mp4s organized as N10/<suite>/...
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

## Verified architecture (cross-checked against `model.safetensors`)

Every dimension below was read directly from
`<ckpt_dir>/model.safetensors` of the `pi05_libero_upstream` checkpoint
(not inferred from the Orbax `_METADATA` sharded shapes). Cross-checked
2026-06-07.

| Component | Shape | Source (safetensors tensor) |
|---|---|---|
| **SigLIP layers** | 27 | counted: `vision_tower.…encoder.layers.{0..26}` |
| **SigLIP hidden** | 1152 | `vision_model.encoder.layers.0.layer_norm1.weight` = (1152,) |
| **SigLIP MLP** | 4304 | `vision_model.encoder.layers.0.mlp.fc1.weight` = (4304, 1152) |
| **SigLIP heads × head_dim** | 16 × 72 | `self_attn.q_proj.weight` = (1152, 1152); 1152/16 = 72 |
| **SigLIP patch** | 14 × 14 × 3 → 1152 | `embeddings.patch_embedding.weight` = (1152, 3, 14, 14) |
| **SigLIP pos_embedding** | (256, 1152) | `embeddings.position_embedding.weight`; 256 = (224/14)² |
| **Multi-modal projector** | 1152 → 2048 | `multi_modal_projector.linear.weight` = (2048, 1152) |
| **Gemma VLM layers** | 18 | counted: `paligemma.…language_model.layers.{0..17}` |
| **Gemma VLM hidden** | 2048 | `language_model.layers.0.input_layernorm.weight` = (2048,) |
| **Gemma VLM Q-heads** | 8 | `self_attn.q_proj.weight` = (2048, 2048); 2048 / head_dim 256 = 8 |
| **Gemma VLM KV-heads** | 1 (MQA) | `self_attn.k_proj.weight` = (256, 2048); 256 / 256 = 1 |
| **Gemma VLM head_dim** | 256 | implied by k/v_proj output dim = 256 |
| **Gemma VLM MLP** | 16384 | `mlp.gate_proj.weight` = (16384, 2048) |
| **Vocab (joint)** | 257152 | `paligemma.lm_head.weight` = (257152, 2048); `gemma_expert.lm_head.weight` = (257152, 1024) |
| **Expert layers** | 18 | counted: `gemma_expert.model.layers.{0..17}` |
| **Expert hidden** | 1024 | `gemma_expert.layers.0.mlp.gate_proj.weight` = (4096, 1024) |
| **Expert Q-heads × head_dim** | 8 × 256 | `self_attn.q_proj.weight` = (2048, 1024); 2048 = 8 × 256 |
| **Expert KV-heads** | 1 (MQA) | `self_attn.k_proj.weight` = (256, 1024) |
| **Expert MLP** | 4096 | `mlp.gate_proj.weight` = (4096, 1024) |
| **Expert adaRMSNorm dense** | 1024 → 3072 | `input_layernorm.dense.weight` = (3072, 1024); 3072 = 3 × hidden (scale+shift+gate) |
| **Action in_proj** | 32 → 1024 | `action_in_proj.weight` = (1024, 32) |
| **Action out_proj** | 1024 → 32 | `action_out_proj.weight` = (32, 1024) |
| **Time MLP in** | 1024 → 1024 | `time_mlp_in.weight` = (1024, 1024) |
| **Time MLP out** | 1024 → 1024 | `time_mlp_out.weight` = (1024, 1024) |

Note: the expert is "Gemma-300M-sized" (18L, 1024H, 4096 MLP) but with
`head_dim=256` to **match the VLM** — not stock Gemma-300M's head_dim=64.
This is required because pi0.5 cross-attention reuses the VLM's KV
cache; expert Q heads must share `head_dim` with VLM K/V. Enforced by an
assertion in `PaliGemmaConfig.__post_init__`.

### Real action / state dims (from `norm_stats.json`)

| Tensor | Real dim | Padded dim (in config) | Notes |
|---|---:|---:|---|
| state | 8 | 32 | joint angles + EE pose; pi05 state-in-prompt mode bin-encodes into language tokens, so the 32 padding is unused |
| actions | 7 | 32 | 6-DoF Cartesian + gripper; padded to 32 for tile alignment |

### Upstream openpi source references

All training-time defaults verified against the public `Physical-Intelligence/openpi` repo (no auth required for source files; only the HF checkpoint mirror is gated):

| File | What it gives us |
|---|---|
| [`src/openpi/training/config.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py) — `pi05_libero` entry | `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` + `LeRobotLiberoDataConfig(repo_id="physical-intelligence/libero")` |
| [`src/openpi/models/pi0_config.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py) — `Pi0Config` | Base defaults: `action_dim=32`, `action_horizon=50` (overridden), `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"`, `max_token_len=200` when `pi05=True` |
| [`src/openpi/models/model.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py) | `IMAGE_RESOLUTION = (224, 224)` |
| `Pi0Config.inputs_spec(...)` | Hardcoded 3-camera dict: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`, each `(B, 224, 224, 3)` |
| [`gs://openpi-assets/checkpoints/pi05_libero/`](https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/) | Authoritative Orbax checkpoint (16 files, ~7.2 GB). `assets/.../norm_stats.json` is public; `params/` is the JAX checkpoint that gets converted to the HF safetensors mirror. |
| [`huggingface.co/openpi/pi05_libero`](https://huggingface.co/openpi/pi05_libero) | Safetensors mirror (gated). Source of `model.safetensors` + minimal `config.json` we use. |

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

## Dtype mapping

Per-stage map of weight / output / compute-config dtypes in the pi0.5 TTNN
pipeline. The `Source commit` column flags ops that were touched by the
bf8 conversion work and reflects the current state — not the full history.
Three commits contributed (all on `sdawle/dvartanians/pi0.5_openpi_upstream`):

- `8ef91d7fe60` — pi0.5 TTNN: bf8_b for SigLIP attention weights, VLM
  expert QKV, and (initially) expert o_proj/MLP outputs.
- `c0876acc212` — pi0.5 TTNN: bf8_b for SigLIP patch conv output and
  (initially) the four pi0.5 suffix linear outputs.
- `df531eeb9d6` — pi0.5 TTNN: SigLIP biases bf16 → bf8_b; revert the
  session-flipped activation outputs (Gemma o_proj/MLP, suffix linears,
  SigLIP patch conv) back to bf16 after the LIBERO 800-episode sweep
  showed a 1-2 pp regression. Weights stay bf8_b.

See `models/experimental/pi0_5/tt/{ttnn_siglip,ttnn_paligemma,ttnn_gemma,ttnn_suffix,ttnn_pi0_5_model}.py` for the live code.

### Inputs (`ttnn_pi0_5_model.py`)

| Tensor | Dtype | Source commit |
|---|---|---|
| Images / state / x_t (initial noise) | `bfloat16` | |
| Lang tokens | `uint32` | |
| Pre-computed `adarms_cond` + per-(step, layer) modulations | `bfloat16` | |

### SigLIP encoder — `ttnn_siglip.py` · 27 layers

| Op | Weight | Output | Source commit |
|---|---|---|---|
| Patch conv weight | `bfloat16` | — | |
| Patch conv output | — | `bfloat16` | `df531eeb9d6` (reverted from bf8_b at line 326) |
| Attention QKV (fused) weight | `bfloat8_b` | — | `8ef91d7fe60` (lines 410/413/416) |
| Attention QKV output | — | `bfloat8_b` | |
| Attention QKV biases (bq/bk/bv) | `bfloat8_b` | — | `df531eeb9d6` (lines 427-429) |
| Attention `out_proj` weight | `bfloat8_b` | — | `8ef91d7fe60` (line 438) |
| Attention `out_proj` output | — | `bfloat8_b` | |
| Attention `out_proj` bias (`bo`) | `bfloat8_b` | — | `df531eeb9d6` (line 445) |
| MLP `fc1` / `fc2` weight | `bfloat8_b` | `bfloat8_b` | |
| MLP `fc1` / `fc2` bias | `bfloat8_b` | — | `df531eeb9d6` (lines 785, 805) |
| Compute kernel | HiFi2, `fp32_dest_acc_en=True`, `packer_l1_acc=True` | | |

### PaliGemma VLM prefill — Gemma-2B · 18 blocks · w=2048

| Op | Weight | Output | Source commit |
|---|---|---|---|
| `embed_tokens` / RMSNorm / RoPE cos+sin | `bfloat16` | — | |
| QKV fused | `bfloat8_b` | `bfloat8_b` | |
| QKV / attention biases | `bfloat16` | — | (not flipped; only SigLIP biases were) |
| KV cache | `bfloat16` | — | intentional (hot-read path) |
| `o_proj` | `bfloat8_b` | `bfloat16` | weight `8ef91d7fe60`; output reverted by `df531eeb9d6` |
| MLP gate/up/down | `bfloat8_b` | `bfloat16` | weight `8ef91d7fe60` (shared `GemmaMLPTTNN.to_ttnn`); output reverted by `df531eeb9d6` |
| Compute kernel | HiFi2, `fp32_dest_acc_en=False`, `packer_l1_acc=True` | | |

### Action expert with adaRMS — Gemma-300M · 18 blocks · w=1024

| Op | Weight | Output | Source commit |
|---|---|---|---|
| QKV fused | `bfloat8_b` | `bfloat8_b` | `8ef91d7fe60` (ttnn_paligemma.py lines 262/268/274) |
| `o_proj` | `bfloat8_b` | `bfloat16` | shared `GemmaAttentionTTNN`; output reverted by `df531eeb9d6` |
| MLP gate/up/down | `bfloat8_b` | `bfloat16` | shared `GemmaMLPTTNN`; output reverted by `df531eeb9d6` |
| adaRMS modulation (precomputed scale/shift/gate per step,layer) | `bfloat16` | `bfloat16` | |
| Sharded RMSNorm compute kernel | HiFi2, `fp32_dest_acc_en=False`, `packer_l1_acc=True` (DST budget) | | |

### Suffix embedding — `ttnn_suffix.py`

| Op | Weight | Output | Source commit |
|---|---|---|---|
| `action_in_proj` | `bfloat8_b` | `bfloat16` (implicit) | output reverted by `df531eeb9d6` |
| `time_mlp_in` | `bfloat8_b` | `bfloat16` (implicit) | output reverted by `df531eeb9d6` |
| `time_mlp_out` | `bfloat8_b` | `bfloat16` (implicit) | output reverted by `df531eeb9d6` |
| `action_out_proj` | `bfloat8_b` | `bfloat16` (implicit) | output reverted by `df531eeb9d6` |
| sincos(t) | — | `float32` on device → cast to `bfloat16` | |
| `adarms_cond` (final) | — | `bfloat16` | |

### Denoise loop — `ttnn_pi0_5_model.py::sample_actions`

| Item | Dtype | Source commit |
|---|---|---|
| In-loop activations / x_t | `bfloat16` | (intentionally fragile to bf8 — opt-in fp32 via `PI0_DENOISE_FP32=1`) |
| `dt` | Python float (velocity scaled via `ttnn.mul(velocity, dt)`) | |

### Output

| Step | Dtype | Source commit |
|---|---|---|
| Final x_t | `bfloat16` | |
| Sliced to logical `action_horizon` → `ttnn.to_torch` | host tensor | |

### LIBERO success rate (upstream pi05_libero, 400 episodes per N, replan=5)

Tracking changes across the bf8 conversion effort, against the user's
pre-bf8 baseline (388 episodes / suite × 4 suites = 800 total):

| Stage | N=10 | N=5 | Combined |
|---|---|---|---|
| Pre-bf8 baseline | 394/400 (98.5%) | 394/400 (98.5%) | 788/800 (98.5%) |
| `8ef91d7fe60` + `c0876acc212` (all weights + outputs bf8) | 390/400 (97.5%) | 387/400 (96.75%) | 777/800 (97.13%) |
| `df531eeb9d6` (current — weights+biases bf8, session outputs reverted) | 389/400 (97.25%) | 387/400 (96.75%) | 776/800 (97.0%) |

`libero_10` task 8 is the recurring loss (4/10 at N=10, 6/10 at N=5 in
the current config) — it has the longest horizon and frequently hits the
env step cap.

### Notes

- The previous `Expert QKV bf16 / VLM QKV bf8_b` and `SigLIP attn bf16 weights / MLP bf8_b weights` asymmetries are gone — both are now uniformly `bfloat8_b` after `8ef91d7fe60`.
- KV cache remains `bfloat16` intentionally; it is read on every expert step and the precision was preserved as a hot-path concession.
- `fp32_dest_acc_en` is `True` for SigLIP / SDPA and `False` for Gemma matmuls + sharded LN. The sharded LN needs the DST budget for the 8×2 sharding.
- Only the **SigLIP** biases were flipped to `bfloat8_b` in `df531eeb9d6`; PaliGemma / Gemma biases (`ttnn_paligemma.py:230/310/592/636`) remain `bfloat16`.
- More bf8 is not strictly better — see the LIBERO regression discussion in `[[pi0_5 accuracy levers]]`. Always re-run a LIBERO sweep before committing dtype flips, not just PCC.

---

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
