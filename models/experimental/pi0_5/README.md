# PI0.5 (`pi0_5`) — Tenstorrent

End-to-end TTNN implementation of the **π₀.₅** (PI0.5) vision-language-action policy on Blackhole, with a PyTorch reference, LIBERO simulator integration, and a real-weights trace+2CQ perf path at **~31 ms / chunk** on a 1×8 Blackhole mesh (**~42 ms** single-chip) — measured against the upstream `pi05_libero` checkpoint (10-action chunks, 5 denoise steps, 3 cameras).

The supported multi-chip path is the **1×8 single-mesh pipeline** (SigLIP DP +
prefill TP=8 + replicated denoise on one 1×8 Blackhole mesh) — see
[Multi-chip pipelines](#multi-chip-pipelines-tttt_bh_glx).

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
│  │  │  Images   │   │   Language    │  │   │           │ Noisy  │     ││
│  │  │  (224x224)│   │ (200 tokens,  │  │   │           │Actions │     ││
│  │  │  (3 views)│   │  task prompt) │  │   │           │(10, 32)│     ││
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
│                       │     (5 denoising steps)      │                 │
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
│                        │     [batch=1, 10, 32]    │                    │
│                        └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **Shared Attention**: VLM and Expert share K,V tensors (concatenated), but have separate Q and MLPs (same as PI0)
- **AdaRMSNorm in Expert**: each layer reads `(scale, shift, gate)` from a Dense projection of `adarms_cond` (time-derived). `out = RMSNorm(x)·(1+scale) + shift`; residual is gated.
- **Language prompt**: the default `pi05_libero` checkpoint (`discrete_state_input=False`) feeds the task description only — there is no separate state token in the suffix and no state projection weight in either expert. (A checkpoint trained with `discrete_state_input=True` instead discretizes the 8-dim robot state into 256 bins appended to the prompt; select it at rollout with `--state-in-prompt true`.)
- **Flow Matching**: same Euler integration as PI0; **10** denoising steps (openpi training default) from N(0,I) → actions. The perf-tuned path uses **5** (validated PCC-equal on LIBERO, ~half the device time).
- **Dual Experts**: VLM (2B) processes images + language; Expert (300M, adaRMS) processes only the action tokens.

### What differs from PI0

| Component        | PI0                                              | **PI0.5**                                                     |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Suffix tokens    | `[state_token, action_0, …, action_{H-1}]`       | `[action_0, …, action_{H-1}]` (state encoded into lang tokens)|
| Time injection   | concat(action, sincos(t)) → 2-layer MLP, fused   | sincos(t) → MLP → `adarms_cond` (fed to adaRMSNorm)           |
| Expert RMSNorm   | Plain RMSNorm                                    | adaRMSNorm: `normed * (1+scale) + shift`, with gated residual |
| `max_token_len`  | 48                                               | 200                                                           |
| State input      | continuous (state_proj)                          | none by default (`discrete_state_input=False`); optionally 256-bin discretized into the language prompt |

Everything else (SigLIP-27, Gemma-2B VLM, Gemma-300M expert, flow-matching denoising with Euler integration, KV-cache prefill of the prefix) follows the openpi/lerobot reference.

---

## Directory layout

```
pi0_5/
├── common/
│   ├── configs.py            # GemmaConfig, SigLIPConfig, PaliGemmaConfig, ...
│   ├── weight_loader.py      # PI0WeightLoader, categorize_weights, Pi0_5WeightLoader
│   ├── checkpoint_meta.py    # action_horizon_from_checkpoint (reads config.json)
│   ├── prod_env.py           # apply_production_env_defaults() loader
│   └── pi05_production.env   # validated production perf-flag defaults
├── reference/                    # PyTorch reference (PCC + libero pytorch backend)
│   ├── torch_denoise.py
│   ├── torch_gemma.py        # GemmaAttn/MLP/Block + AdaRMSGemmaBlock
│   ├── torch_paligemma.py    # PaliGemmaBackbone + Pi0_5PaliGemmaBackbone
│   ├── torch_prefix.py
│   ├── torch_siglip.py
│   ├── torch_suffix.py       # SuffixEmbedding + Pi0_5SuffixEmbedding
│   └── torch_pi0_5_model.py  # Pi0_5Model
├── tt/                       # TTNN implementation
│   ├── ttnn_{common,gemma,paligemma,prefix,siglip,suffix}.py
│   ├── ttnn_pi0_5_model.py   # Pi0_5ModelTTNN (single-chip)
│   └── tt_bh_glx/            # 1×8 single-mesh pipeline (see below)
│       ├── pipeline_1x8.py   #   Pi0_5GLX1x8Pipeline (supported multi-chip path)
│       ├── heads.py          #   _PrefillHead / _DenoiseHead (shared, standalone)
│       ├── stage_prefill_tp4.py, *_slice.py, stages.py, mesh_setup.py
├── libero_sim/               # LIBERO simulator rollout (see libero_sim/README.md)
│   ├── libero_rollout.py     #   checkpoint → policy → success rate / videos
│   └── async_rollout.py
├── tests/
│   ├── pcc/                  # Reference-vs-spec correctness
│   └── perf/                 # Latency / throughput on Blackhole
└── weights/                  # checkpoints (not tracked; see weights/README.md)
    └── download_pi05_libero.py  # download + prepare + verify the upstream checkpoint
```

---

## Quickstart

Set once (get the checkpoint via [`weights/download_pi05_libero.py`](weights/README.md)):

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
```

### TTNN — e2e trace + 2CQ perf (Blackhole)

The tests self-select devices (single-chip defaults to chip 9; the 1×8 test to
chips 8–15) — override with `TT_VISIBLE_DEVICES` if needed. Two knobs vary the
workload (defaults from `pi05_production.env`):

- `PI0_NUM_CAMERAS` — 2 or 3 cameras (3 = training spec; 2 also set `PI0_VLM_CHUNK_SIZE=768`).
- `PI05_NUM_DENOISE_STEPS` — denoise steps (5 = perf-tuned, 10 = training default).

```bash
source models/experimental/pi0_5/common/pi05_production.env   # perf flags (tests auto-apply too)

# Single BH chip
PI0_NUM_CAMERAS=3 PI05_NUM_DENOISE_STEPS=5 \
  python_env/bin/pytest -sq models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace_2cq.py

# 8 BH chips (1×8 mesh)
PI0_NUM_CAMERAS=3 PI05_NUM_DENOISE_STEPS=5 \
  python_env/bin/pytest -sq models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8.py
```

### LIBERO sim — task-success rollout

Full setup + all flags in [`libero_sim/README.md`](libero_sim/README.md). Prefix the
LIBERO env vars (`PI0_TOKENIZER_PATH`, `LIBERO_REPO_PATH`, `MUJOCO_GL=osmesa`) and pick
`--backend ttnn` (single chip) or `ttnn_1x8` (`TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`):

```bash
ROLL="models/experimental/pi0_5/libero_sim/libero_rollout.py --checkpoint $PI05_CHECKPOINT_DIR \
  --backend ttnn --steps-sweep 5 --replan-steps 5 \
  --suites libero_spatial libero_object libero_goal libero_10 --task-range 0 9"

# Quick check — 40 episodes (1 init/task × 10 tasks × 4 suites)
python_env/bin/python -u $ROLL --num-episodes 1

# Full run — 400 episodes (10 init/task × 10 tasks × 4 suites)
python_env/bin/python -u $ROLL --num-episodes 10
```

---

## Tests

Skipped automatically if the checkpoint (`$PI05_CHECKPOINT_DIR`) is missing. All
commands assume:

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
```

### PCC tests

Compare TTNN vs the PyTorch reference on real upstream weights, gated at **PCC ≥ 0.99**.

**Single BH chip**
```bash
# per-block vs torch + single-chip e2e
python_env/bin/pytest -sq \
  models/experimental/pi0_5/tests/pcc/test_pcc_siglip_vs_torch.py \
  models/experimental/pi0_5/tests/pcc/test_pcc_paligemma_vs_torch.py \
  models/experimental/pi0_5/tests/pcc/test_pcc_prefix_vs_torch.py \
  models/experimental/pi0_5/tests/pcc/test_pcc_suffix_vs_torch.py \
  models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model_libero.py
# single-chip (tp1) stages
python_env/bin/pytest -sq \
  "models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py::test_prefill_tp1_pcc" \
  "models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py::test_vision_tp1_pcc"
```

**8 BH chips (1×8 mesh)**
```bash
PI05_E2E_PCC=1 python_env/bin/pytest -sq models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_1x8.py
```

Results (upstream pi05_libero, ≥ 0.99 bar):

| test | chips | PCC |
|---|---|---|
| SigLIP vision tower | 1 | 0.9976 |
| PaliGemma embed_image / VLM block | 1 | 0.9997 / 0.9973 |
| prefix / suffix embedding | 1 | 0.9999 / 0.9997 |
| single-chip e2e | 1 | 0.9963 |
| single-chip stages (prefill / vision tp1) | 1 | 0.9934 / 0.9997 |
| 1×8 vision / prefill / e2e | 8 | 0.9997 / 0.9946 / 0.9965 |

E2E PCC is mildly seed-sensitive (the 5-step flow-matching ODE amplifies bf16 drift
per initial-noise pattern), so the bar keeps headroom above the ~0.996 typical value.

### Perf tests

Trace + 2CQ (the canonical "fast" path). Vary the workload with two env vars
(defaults from `pi05_production.env`):

- `PI0_NUM_CAMERAS` — `2` or `3` cameras (`3` = training spec; `2` also set `PI0_VLM_CHUNK_SIZE=768`).
- `PI05_NUM_DENOISE_STEPS` — denoise steps (`5` = perf-tuned, `10` = training default).

**Single BH chip**
```bash
PI0_NUM_CAMERAS=3 PI05_NUM_DENOISE_STEPS=5 \
  python_env/bin/pytest -sq models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace_2cq.py
```

**8 BH chips (1×8 mesh)**
```bash
PI0_NUM_CAMERAS=3 PI05_NUM_DENOISE_STEPS=5 \
  python_env/bin/pytest -sq models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8.py
```

Results (trace + 2CQ, N=5 — per-chunk ms · throughput):

| | 2 cameras | 3 cameras |
|---|---|---|
| **Single BH chip** | 36.6 ms · 273 act/s | 42.3 ms · 236 act/s |
| **8 BH chips (1×8)** | 29.0 ms · 345 act/s | 31.2 ms · 320 act/s |

10-action chunks (throughput = 10 / per-chunk). The 1×8 mesh absorbs the extra camera
far better (+2.2 ms for 2→3 cam vs +5.7 ms single-chip). The SigLIP encoder runs in L1
block-sharded layout (12×8 grid); `PI0_SIGLIP_BS=0` reverts to interleaved-LN with no
rebuild. Per-stage / CCL breakdown: `test_perf_tt_bh_glx_1x8.py::test_perf_1x8_traced_staged`.

---

## Multi-chip pipelines (`tt/tt_bh_glx/`)

Everything above runs on a **single Blackhole chip**. The supported multi-chip
path is the **1×8 single-mesh pipeline** (`pipeline_1x8.py`,
`Pi0_5GLX1x8Pipeline`): SigLIP DP + Prefill TP=8 + replicated denoise, all on one
1×8 mesh, with on-device CCL for cross-stage handoff (no host bounce, no fabric
sockets). This is what the `test_*_tt_bh_glx_1x8` PCC/perf tests and the
`--backend ttnn_1x8` LIBERO rollout exercise.

- Open the mesh with `mesh_setup.open_prefill_tp4_mesh(tp=8, num_command_queues=2)`
  (chips 8–15 on this box).
- The 1×8-specific flags (`PI0_TP=8`, `PI0_TP4_ATTN_HEADPAR=1`, `PI0_MLP_BS=1`,
  `PI0_MLP_FUSED_RS=0`) are auto-applied by the 1×8 test files.
- Exercised by `tests/pcc/test_pcc_tt_bh_glx_1x8.py` and
  `tests/perf/test_perf_tt_bh_glx_1x8.py` (see [Tests](#tests)).

`pipeline_1x8.py` is **self-contained**: its only shared building blocks are the small
`_PrefillHead` / `_DenoiseHead` classes in `heads.py` and the `*_slice.py` stage slices —
no dependency on any legacy driver. The older 28-chip host-bounce pipeline
(`pipeline.py` / `Pi0_5GLXPipeline`) and its stage/transport/migration machinery
(`stage_{vision,prefill,denoise}.py`, `transport.py`, `kv_migration.py`,
`_l1_migration.py`) plus the `--backend ttnn_glx` path were removed in the cleanup;
`mesh_setup.open_galaxy_mesh` is retained for reference.

---

## LIBERO simulator rollout

End-to-end benchmark on the four LIBERO suites (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`), 10 tasks each.

### One-time setup

```bash
export PI05_BASE=$HOME/pi05_cache        # any writable dir

# 1. PaliGemma tokenizer (~4 MB, public)
curl -L -o $PI05_BASE/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# 2. Checkpoint: upstream openpi pi05_libero (torch/safetensors). Downloads +
#    fills config.json / norm_stats + verifies via the loader. Gated repo →
#    `huggingface-cli login` first. See weights/README.md.
huggingface-cli login
python_env/bin/python models/experimental/pi0_5/weights/download_pi05_libero.py \
  --out $PI05_BASE/pi05_libero_upstream
export PI05_CHECKPOINT_DIR=$PI05_BASE/pi05_libero_upstream

# 3. LIBERO from source (the PyPI package is broken)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $PI05_BASE/libero_repo

# 4. System packages for headless MuJoCo render
sudo apt install -y libosmesa6 libegl1-mesa xvfb ffmpeg

# 5. Python deps into python_env (uv-managed). robosuite 1.4.0 is REQUIRED
#    (1.5.x breaks libero 0.1.0's import path); do NOT pin numpy (<2 downgrades ttnn).
export VIRTUAL_ENV=$PWD/python_env
uv pip install "robosuite==1.4.0" mujoco bddl easydict cloudpickle gym imageio-ffmpeg
uv pip install --no-deps -e $PI05_BASE/libero_repo
```

### Run

```bash
source models/experimental/pi0_5/common/pi05_production.env      # perf flags + checkpoint path

PI0_TOKENIZER_PATH=$PI05_BASE/paligemma_tokenizer.model \
LIBERO_REPO_PATH=$PI05_BASE/libero_repo \
MUJOCO_GL=osmesa TT_METAL_HOME=$PWD PYTHONPATH=$PWD:$PI05_BASE/libero_repo \
python_env/bin/python -u models/experimental/pi0_5/libero_sim/libero_rollout.py \
  --checkpoint $PI05_CHECKPOINT_DIR \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task-range 0 9 --num-episodes 1 --steps-sweep 5 \
  --backend ttnn --replan-steps 5
# → 40 episodes (1 init/task × 4 suites). Add --video-dir <dir> for per-episode mp4s;
#   scale up with --num-episodes N (up to 50 canonical inits/task).
```

Machine-specific env vars (not in `pi05_production.env`): `PI0_TOKENIZER_PATH`,
`LIBERO_REPO_PATH`, `MUJOCO_GL=osmesa`, and `TT_METAL_CACHE` if `$HOME/.cache` is a
dangling symlink.

Key flags: `--backend {ttnn | ttnn_1x8 | pytorch}` · `--steps-sweep 5` (our path) ·
`--action-horizon 10` / `--state-in-prompt false` (upstream defaults) ·
`--replan-steps 5` · `--num-episodes` · `--suites` · `--task-range` · `--max-steps`
(per-suite defaults: spatial=220, object=280, goal=300, libero_10=520).

### Per-chunk latency (N=5)

Untraced rollout ≈ **250 ms/chunk** (host↔device transfer bound); the **trace + 2CQ**
path is **~42 ms single-chip / ~31 ms on 1×8** (see [Perf tests](#perf-tests-blackhole)).
Task success is in the [400-episode/suite sweep](#libero-success-rate-upstream-pi05_libero-400-episodes-per-n-replan5) below.

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

Weights and matmul activations are **`bfloat8_b`** across SigLIP, the VLM, and the
action expert; activation **outputs are `bfloat16`** (the bf8-output flips were
reverted after an 800-episode LIBERO sweep showed a 1–2 pp regression — weights stay
bf8). The **KV cache stays `bfloat16`** (hot-read path). Compute kernels are HiFi2
with `fp32_dest_acc_en=True` for SigLIP/SDPA and `False` for the Gemma matmuls +
sharded LN. Live code:
`tt/{ttnn_siglip,ttnn_paligemma,ttnn_gemma,ttnn_suffix,ttnn_pi0_5_model}.py`.

| Stage | Weights / biases | Matmul output | Notes |
|---|---|---|---|
| Inputs | — | images/state/x_t `bf16`; lang tokens `uint32` | `adarms_cond` precomputed `bf16` |
| SigLIP · 27 layers | `bf8_b` | attn + MLP `bf8_b` | patch-conv weight `bf16`; `fp32_dest_acc_en=True` |
| VLM Gemma-2B · 18 | `bf8_b` | `bf16` | KV cache + biases `bf16`; `fp32_dest_acc_en=False` |
| Expert Gemma-300M · 18 | `bf8_b` | `bf16` | adaRMS modulation `bf16`; sharded RMSNorm |
| Suffix · 4 linears | `bf8_b` | `bf16` | sincos(t) `fp32`→`bf16` |
| Denoise loop | — | x_t `bf16` | intentionally `bf16` (opt-in fp32 via `PI0_DENOISE_FP32=1`) |

> More bf8 is not strictly better — always re-run a LIBERO sweep (not just PCC)
> before flipping dtypes. See `[[pi0_5 accuracy levers]]`.

### LIBERO success rate (upstream pi05_libero, 400 episodes/suite × 4 suites, N=5, replan=5)

Tracking changes across the bf8 conversion effort (1600 episodes total per row):

| Stage | N=5 |
|---|---|
| Pre-bf8 baseline | 394/400 (98.5%) |
| `8ef91d7fe60` + `c0876acc212` (all weights + outputs bf8) | 387/400 (96.75%) |
| `df531eeb9d6` (current — weights+biases bf8, session outputs reverted) | 387/400 (96.75%) |

`libero_10` task 8 is the recurring loss (6/10 in the current config) — it has the
longest horizon and frequently hits the env step cap.

---

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
