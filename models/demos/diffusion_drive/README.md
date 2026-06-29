# DiffusionDrive Autonomous Driving Planner (TTNN)

**Platform:** Wormhole N300s

## Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [TTNN Optimization Techniques](#3-ttnn-optimization-techniques)
4. [Bring-Up Stages](#4-bring-up-stages)
5. [Performance](#5-performance)
6. [Setup](#6-setup)
7. [Running Tests](#7-running-tests)
8. [Usage](#8-usage)
9. [NavSim PDM Evaluation](#9-navsim-pdm-evaluation)

---

## 1. Overview

[DiffusionDrive](https://arxiv.org/abs/2411.01799) (CVPR 2025, Shao et al.) is a
diffusion-based end-to-end autonomous driving planner that won the NavSim challenge.
It generates K=20 candidate trajectories in a single forward pass via a 2-step DDIM
denoiser conditioned on a camera+LiDAR BEV feature.

This bring-up ports DiffusionDrive to Wormhole N300s in stages, replacing PyTorch
ops with native TTNN kernels one subsystem at a time and validating PCC ≥ 0.99 at
each step.

**Architecture summary**

| Component | Shape in / out | Description |
|---|---|---|
| Image encoder | (B,3,256,1024) → (B,512,8,32) | ResNet-34, 4 stages, BN-folded |
| LiDAR encoder | (B,1,256,256) → (B,512,8,8) | ResNet-34, 4 stages, BN-folded |
| GPT fusion | — | 4-scale cross-modal transformer |
| FPN | (B,512,8,8) → (B,64,64,64) | 3-level top-down, bilinear upsample |
| Perception decoder | (B,65,512) → (B,31,256) | 3-layer TransformerDecoder |
| Trajectory head | (B,31,256) → trajectory, scores | DDIM 2-step, K=20, T=8 |
| **Outputs** | — | `trajectory` (B×8×3), `scores` (B×20) |

Model checkpoint: `hustvl/DiffusionDrive` — 60 M parameters.

---

## 2. Architecture

### Data flow

```
Camera (B,3,256,1024)
  └─► conv1+bn1+relu+maxpool (TTNN stem)
       └─► layer1…layer4 (TTNN — 16 BasicBlocks) ──────────────────────┐
                                                                         │
LiDAR (B,1,256,256)                                                      │
  └─► conv1+bn1+relu+maxpool (TTNN stem)                       GPT fusion
       └─► layer1…layer4 (TTNN — 16 BasicBlocks) ──────────── ×4 scales │
                                              (TTNN ← avg_pool2d + 1×1 linear +
                                               2-layer GPT + ttnn.upsample)
                                                                         │
                                                              lidar_feats (B,512,8,8)
                                                                         │
                                                    ┌────────────────────┤
                                                    │         bev_feature│
                                          FPN _top_down                  │
                                  c5_conv 1×1  (TTNN)                   │
                                  upsample 2×  (TTNN ttnn.upsample)     │
                                  up_conv5 3×3 (TTNN)                   │
                                  upsample to 64×64 (TTNN)              │
                                  up_conv4 3×3 (TTNN)                   │
                                       │                                 │
                                bev_upscale (B,64,64,64)                │
                                                                         │
                           _bev_downscale (TTNN 1×1 conv)              │
                           _status_encoding (TTNN Linear)              │
                           _keyval_embedding (host add — glue)          │
                           F.interpolate + cat bev_upscale              │
                           bev_proj: Linear(320,256)+ReLU+LN (TTNN)    │
                                       │                                 │
                           _tf_decoder (TTNN TransformerDecoder)       │
                                       │                                 │
                           TrajectoryHead DDIM ◄──────────── bev_feature┘
                           (TTNN denoiser + ttnn.grid_sample;
                            DDIM schedule arithmetic on host)
                                       │
                             trajectory (B,8,3)  scores (B,20)
```

### TTNN vs PyTorch per submodule (after `build_stage3_7`)

Every **weight-bearing** op runs on TTNN. The host residue is non-weight scalar
glue (enumerated in the table below).

| Submodule | Execution | Stage | Notes |
|---|---|---|---|
| ResNet-34 stems (conv1+bn1+relu+maxpool) ×2 | **TTNN** | 3.6 | `TtnnStem`: BN-fold conv + `ttnn.max_pool2d` |
| ResNet-34 BasicBlocks ×32 | **TTNN** | 2 | BN-fold enables weight-only conv |
| GPT cross-modal fusion ×4 | **TTNN** | 3.6 | `avg_pool2d` + 1×1 `linear` channel proj + GPT (LN/attn/MLP) + bilinear `upsample` + residual; integer ratios at production res |
| FPN conv ×3 + bilinear upsample ×2 | **TTNN** | 3 | `ttnn.upsample(bilinear)` matches `align_corners=False` to PCC ≥ 0.99999 |
| `_bev_downscale`, `_status_encoding`, `bev_proj` | **TTNN** | 3.4 | 1×1 conv + Linears |
| Perception TransformerDecoder ×3 | **TTNN** | 3.4 | SDPA + FFN + LN |
| TrajectoryHead DDIM denoiser (incl. `grid_sample`) | **TTNN** | 3.5 | plan_anchor_encoder, time_mlp, grid-sample cross-attn, 2× MHA, FFN, norms, FiLM, task heads |
| `_agent_head` MLPs | **TTNN** | 3.7 | `_mlp_states`, `_mlp_label` |
| DDIM `scheduler.step`, `gen_sineembed`, norm/denorm, argmax/gather, embedding-add | host (glue) | — | scalar/indexing on ≤320-elt tensors; not kernel-worthy |

---

## 3. TTNN Optimization Techniques

### 3.1 BatchNorm folding into conv weights

`ttnn.conv2d` carries no BatchNorm op — weights must be pre-folded.  For each
`conv + bn` pair the running statistics are absorbed into the conv weights and bias
before any TTNN call:

```python
# Fold: w_folded = w * (gamma / sqrt(var + eps))
#        b_folded = (b - mean) * gamma / sqrt(var + eps) + beta
scale = gamma / torch.sqrt(running_var + eps)          # (C_out,)
w_folded = w * scale.view(-1, 1, 1, 1)
b_folded = (b_conv - running_mean) * scale + beta
```

This eliminates runtime per-channel scale+shift and keeps the weight tensor in
`bfloat16` throughout.  The FPN convolutions have *no* BatchNorm so their weights
are cast to `bfloat16` directly with no folding step.

### 3.2 NHWC tile layout for conv inputs

`ttnn.conv2d` expects the activation in `(1, 1, B·H·W, C)` TILE layout (NHWC
order, flat batch-spatial, tile-padded to 32).  The helpers `_to_ttnn_tile` /
`_from_ttnn_tile` in `tt/ttnn_backbone.py` handle this round-trip:

```python
def _to_ttnn_tile(x, B, H, W, C, device):
    # (B,C,H,W) → (1,1,B*H*W,C) bfloat16 TILE on device
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_flat  = x_nhwc.reshape(1, 1, B * H * W, C).to(torch.bfloat16)
    return ttnn.from_torch(x_flat, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)
```

The inverse permutes back to `(B, C, H_out, W_out)` float32 for the PyTorch
parts of the pipeline.

### 3.3 L1_SMALL allocation for conv2d

`ttnn.conv2d` allocates from the `L1_SMALL` bank.  The default `l1_small_size=0`
leaves the allocator with zero banks, causing every conv to fail with:

```
RuntimeError: Out of Memory: Not enough space to allocate ... bank size is 0 B
```

The fix is to open the device with `l1_small_size=32768` — this is already set in
`tests/conftest.py` and the Usage example below.

### 3.4 FPN bilinear upsample runs on TTNN

The FPN needs two bilinear upsamples (8×8→16×16 and 16×16→64×64).
`ttnn.upsample(mode="bilinear")` **is** available in this build (integer scale
factors only — the FPN's ×2 and ×4 both qualify) and matches torch
`F.interpolate(mode="bilinear", align_corners=False)` to PCC ≥ 0.99999 on both
steps.  `TtnnFPN` therefore runs both upsamples on-device; the earlier
`F.interpolate` fallback has been removed.  (An earlier note here claimed
`ttnn.upsample` was nearest-only — that is no longer accurate.)

### 3.5 DDIM noise seeding — a non-obvious PCC pitfall

This model is **not deterministic** by default.

`TrajectoryHead._forward_test` calls `torch.randn(img.shape, device=device)` at
the start of every forward pass to sample DDIM noise.  There is no seed argument —
it draws directly from the global PyTorch RNG.

When comparing a PyTorch reference run to a TTNN run, the two calls consume
different RNG states unless the seed is explicitly reset before each one.  The
symptom is a plausible-looking but wrong PCC (~0.75 for trajectory, ~0.35 for
scores) even when the backbone and FPN outputs are numerically correct (PCC
≈ 0.9999).

**The fix — reset the same seed before every forward call:**

```python
torch.manual_seed(1234)
with torch.no_grad():
    ref_out = ref_model(features)    # consumes RNG state A

# ... build TTNN model ...

torch.manual_seed(1234)              # reset to same state A
ttnn_out = ttnn_model(features)      # draws identical noise
```

This applies to every PCC test that compares TTNN output against the reference.
Forgetting the second `manual_seed` is the most common failure mode when writing
new full-model tests.

---

## 4. Bring-Up Stages

### Stage summary

| Stage | Scope | TTNN conv ops added | Tests | Commit |
|---|---|---|---|---|
| 0 | Architecture audit, reference model confirmed | 0 | — | — |
| 1 | PyTorch wrapper + BN-fold primitives + BasicBlock PCC | 0 | 14/14 | `857671c0aa` |
| 2 | All 32 ResNet-34 BasicBlock conv layers on TTNN | +70 | 15/15 | `a72716b165` |
| 3 | FPN 3 conv layers on TTNN (`TtnnFPN`) | +3 | 18/18 | `edd70f9e9f` |
| 3.1 | Review fixes: 2-ch DDIM noise (upstream match), FPN bilinear upsample on TTNN, `ttnn.grid_sample` validated, conv-weight caching | +0 conv (+2 upsample) | 24/24 | `4b07970` |
| 3.4 | Perception head on TTNN (`_bev_downscale`, `_status_encoding`, `bev_proj`, 3-layer `_tf_decoder`) | +1 conv | — | `ca36c5b0` |
| 3.5 | DDIM denoiser on TTNN (plan_anchor_encoder, time_mlp, grid-sample cross-attn, 2× MHA, FFN, norms, FiLM, task heads) | — | — | `ca36c5b0` |
| 3.6 | Backbone completion: ResNet stems ×2 + GPT cross-modal fusion ×4 on TTNN (`build_stage3_6`) | +grid/pool/upsample | — | `30cca82a69` |
| 3.7 | Agent head MLPs on TTNN (`build_stage3_7`) — **every weight op now on TTNN** | — | — | `30cca82a69` |

**Current total: 32 PCC tests** (a few skip without the real checkpoint/anchor
assets).  After `build_stage3_7` every
weight-bearing op runs on TTNN; `test_pcc_stage3_6.py` validates the whole
on-device model at production resolution (trajectory PCC 1.0 random / 0.9998
real-checkpoint).  Remaining host code is non-weight scalar glue (enumerated in
the §2 submodule table).

### TTNN ops on-device at Stage 3

```
image_encoder.layer1  — 3 blocks × 2 convs           =  6 ops  (64-ch, 64×256)
image_encoder.layer2  — 4 blocks × 2 convs + 1 downsample =  9 ops  (128-ch, 32×128)
image_encoder.layer3  — 6 blocks × 2 convs + 1 downsample = 13 ops  (256-ch, 16×64)
image_encoder.layer4  — 3 blocks × 2 convs + 1 downsample =  7 ops  (512-ch, 8×32)
                                                 subtotal = 35 ops

lidar_encoder.layer1  — 3 blocks × 2 convs           =  6 ops  (64-ch, 64×64)
lidar_encoder.layer2  — 4 blocks × 2 convs + 1 downsample =  9 ops  (128-ch, 32×32)
lidar_encoder.layer3  — 6 blocks × 2 convs + 1 downsample = 13 ops  (256-ch, 16×16)
lidar_encoder.layer4  — 3 blocks × 2 convs + 1 downsample =  7 ops  (512-ch, 8×8)
                                                 subtotal = 35 ops

FPN c5_conv           — 1×1 conv2d  (512→64, 8×8)           =  1 op
FPN up_conv5          — 3×3 conv2d  (64→64, 16×16)          =  1 op
FPN up_conv4          — 3×3 conv2d  (64→64, 64×64)          =  1 op
                                                               ──────
                                                               73 ops total
```

---

## 5. Performance

Measured on Wormhole N300s, batch=1, full production resolution
(camera 256×1024, LiDAR 256×256), `latent=True`.

### Latency

Every weight-bearing op runs on-device, and the model executes as a **consolidated
graph** rather than a staged sequence of host↔device hops. The TransFuser backbone
runs as one consolidated device-native graph by default — stems → [stage →
fusion] × 4 → FPN chained ttnn→ttnn, eliminating the 8 per-stage host round-trips
(`DD_CONSOLIDATE=0` to opt out) — and `build_stage4` consolidates the perception
head + DDIM decoder on-device as well. Stage 7 then captures that consolidated
backbone loop as a TTNN trace (`compile()` / `execute_compiled()`) and replays it
as a single command: traced-vs-eager trajectory PCC 1.0, with the backbone loop
**1.76×** faster and the full forward **1.34×** (same-process A/B, batch=1,
production resolution; the still-eager FPN/perception/DDIM tail dilutes the loop's
gain).

Absolute forward latency is hardware- and build-dependent, so it isn't pinned
here — measure it on your own setup with the [Profiling](#profiling) snippet below.
End-to-end NavSim eval throughput is gated by host-side navsim CPU rather than the
model forward — see §9.

### Accuracy (PCC vs PyTorch reference)

PCC is Pearson Correlation Coefficient between flattened output tensors.
Both calls use identical DDIM noise (`torch.manual_seed(1234)`).

| Output | Stage 2 PCC | Stage 3 PCC | Threshold |
|---|---|---|---|
| `trajectory` (B×8×3) | 1.000000 | 1.000000 | ≥ 0.99 |
| `scores` (B×20) | 0.999989 | 1.000000 | ≥ 0.99 |

### Profiling

To measure forward-pass latency locally:

```python
import time, torch, ttnn

device = ttnn.open_device(device_id=0, l1_small_size=32768)
# ... build the full ttnn_model: .build_stage2(device) … .build_stage4(device) ...
# (traced path: open with trace_region_size=256*1024*1024, call model.compile() once,
#  then time model.execute_compiled(features) instead of ttnn_model(features))

features = {
    "camera_feature": torch.randn(1, 3, 256, 1024),
    "lidar_feature":  torch.zeros(1, 1, 256, 256),
    "status_feature": torch.zeros(1, 8),
}

# Warm up (populates JIT program cache)
for _ in range(2):
    torch.manual_seed(1234)
    ttnn_model(features)

# Measure
times = []
for _ in range(5):
    torch.manual_seed(1234)
    t0 = time.perf_counter()
    ttnn_model(features)
    times.append((time.perf_counter() - t0) * 1000)

print(f"min={min(times):.0f} ms  avg={sum(times)/len(times):.0f} ms")
ttnn.close_device(device)
```

---

## 6. Setup

```bash
# 1. Create and activate the project virtual environment
source python_env/bin/activate
export PYTHONPATH="${TT_METAL_HOME:-$(pwd)}"   # TT_METAL_HOME = your tt-metal checkout

# 2. Verify device access
python3 -m ttnn.examples.usage.run_op_on_device   # should print a bfloat16 tensor
tt-smi                                             # device info
```

### Assets

Download the NavSim checkpoint and extract the K=20 anchor clusters:

```bash
python models/demos/diffusion_drive/scripts/prepare_assets.py
```

| File | Description |
|---|---|
| `data/diffusiondrive_navsim.pth` | Full model checkpoint (`hustvl/DiffusionDrive`) |
| `data/kmeans_navsim_traj_20.npy` | K-means anchor array, shape (20, 8, 2) |

Tests that require the anchor file skip automatically when it is absent. The
real-checkpoint gates (`test_pcc_checkpoint_accuracy.py`, `test_pcc_trace.py`)
load the trained checkpoint from `$DD_CHECKPOINT_PATH` (falling back to
`$DD_DATA_ROOT/weights/…`, default `/mnt/diffusion-drive`) and skip cleanly when
it is absent. The full eval-asset env-var scheme is in
[§9 NavSim PDM Evaluation](#9-navsim-pdm-evaluation).

---

## 7. Running Tests

```bash
source python_env/bin/activate
export PYTHONPATH="${TT_METAL_HOME:-$(pwd)}"   # TT_METAL_HOME = your tt-metal checkout

# Full suite (32 PCC + 3 sanity = 35 tests; some PCC tests skip without the checkpoint/anchor assets)
python -m pytest models/demos/diffusion_drive/tests/ -v

# PCC tests only (require attached Wormhole device)
python -m pytest models/demos/diffusion_drive/tests/pcc/ -v

# Sanity tests (no device needed)
python -m pytest models/demos/diffusion_drive/tests/sanity/ -v
```

Individual test files (the full set lives in `tests/pcc/`):

```bash
P=models/demos/diffusion_drive/tests/pcc

# No device — BN-fold + DDIM noise-channel correctness
python -m pytest $P/test_pcc_bn_fold.py $P/test_noise_channels.py -v

# Backbone / FPN / consolidated backbone (Stage 2 → 3.6)
python -m pytest $P/test_pcc_resnet_block.py $P/test_pcc_backbone.py \
                 $P/test_pcc_fpn.py $P/test_pcc_backbone_consolidated.py -v

# Full on-device model (Stage 2 → 4) + grid-sample attention
python -m pytest $P/test_pcc_stage2.py $P/test_pcc_stage3.py $P/test_pcc_stage4.py \
                 $P/test_pcc_full_model.py $P/test_pcc_grid_sample.py -v

# Real-checkpoint gates — need $DD_CHECKPOINT_PATH (+ anchors); skip otherwise
python -m pytest $P/test_pcc_checkpoint_accuracy.py $P/test_pcc_trace.py -v
```

---

## 8. Usage

```python
import torch
import ttnn
from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, DiffusionDriveModel
from models.demos.diffusion_drive.tt.config import ModelConfig
from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

# Open device (l1_small_size required for ttnn.conv2d)
device = ttnn.open_device(device_id=0, l1_small_size=32768)

# Build reference model (requires prepare_assets.py)
cfg = DiffusionDriveConfig(
    plan_anchor_path="models/demos/diffusion_drive/data/kmeans_navsim_traj_20.npy"
)
ref_model = DiffusionDriveModel(cfg).eval()

# Build the full on-device model (every weight-bearing op on TTNN)
model_config = ModelConfig()
ttnn_model = (
    TtnnDiffusionDriveModel(ref_model, model_config, device)
    .build_stage2(device)      # ResNet-34 BasicBlocks
    .build_stage3(device)      # + FPN
    .build_stage3_4(device)    # + perception head
    .build_stage3_5(device)    # + DDIM denoiser
    .build_stage3_6(device)    # + ResNet stems + GPT fusion (needs production resolution)
    .build_stage3_7(device)    # + agent-head MLPs
    .build_stage4(device)      # consolidate perception + decoder on-device
)
# Traced fast path: open the device with trace_region_size=256*1024*1024, then call
# model.compile() once and model.execute_compiled(features) per call (see §5).

# Inference — reset seed before each call so DDIM noise is reproducible
features = {
    "camera_feature": torch.randn(1, 3, 256, 1024),
    "lidar_feature":  torch.zeros(1, 1, 256, 256),
    "status_feature": torch.zeros(1, 8),
}
torch.manual_seed(1234)
out = ttnn_model(features)
# out["trajectory"]: (1, 8, 3) — x, y, heading at T=0.5s…4s
# out["scores"]:     (1, 20)   — log-scores for each of K=20 anchor modes

ttnn.close_device(device)
```

### Directory layout

```text
diffusion_drive/
├── README.md
├── data/                        # downloaded assets (gitignored)
│   ├── diffusiondrive_navsim.pth
│   └── kmeans_navsim_traj_20.npy
├── reference/
│   └── model.py                 # PyTorch reference (DiffusionDriveModel)
├── scripts/
│   ├── prepare_assets.py        # download checkpoint + extract anchors
│   ├── ttnn_pdm_server.py       # TTNN inference server (bridge fallback)
│   ├── navsim_inproc/           # default PDM-eval agent (in-process, single env)
│   └── navsim_bridge/           # fallback PDM-eval agent (cross-process socket)
├── tests/
│   ├── conftest.py              # device fixture (l1_small_size=32768) + asset resolution
│   ├── pcc/                     # accuracy tests (require device)
│   └── sanity/                  # NaN/Inf + range checks (no device)
└── tt/
    ├── config.py                # ModelConfig dataclass
    ├── common.py                # BN-fold + weight-preprocessing helpers
    ├── ttnn_resnet34.py         # TtnnBasicBlock (BN-fold + ttnn.conv2d)
    ├── ttnn_backbone.py         # TtnnTransfuserBackbone (stems, BasicBlocks, consolidation + trace)
    ├── ttnn_gpt_fusion.py       # GPT cross-modal fusion — TtnnFuseFeatures
    ├── ttnn_fpn.py              # TtnnFPN
    ├── ttnn_perception.py       # perception-head TTNN drop-ins
    ├── ttnn_grid_sample_attention.py  # GridSampleCrossBEVAttention
    ├── ttnn_trajectory.py       # DDIM denoiser + agent head
    ├── ttnn_consolidated.py     # consolidated perception/decoder forward
    └── ttnn_diffusion_drive.py  # TtnnDiffusionDriveModel (build_stage2…4, compile/execute_compiled)
```

---

## 9. NavSim PDM Evaluation

The headline metric for DiffusionDrive is the **NavSim PDM score** (Predictive
Driver Model), computed by the NavSim devkit's `run_pdm_score.py` over the
`navtest` split (~12 k scenarios). The TTNN model plugs into that harness as a
NavSim *agent*. This section is the end-to-end recipe; the per-transport details
live in two script READMEs:

- [`scripts/navsim_inproc/`](scripts/navsim_inproc/README.md) — **default**:
  single-env, in-process agent (runs the model in the same process as
  `run_pdm_score.py`; no socket).
- [`scripts/navsim_bridge/`](scripts/navsim_bridge/README.md) — fallback:
  cross-process socket bridge (use only if `ttnn` cannot import in the navsim env).

**Validated result:** TTNN PDM ≈ **0.8789** over the full `navtest` (12146
scenarios) — vs the PyTorch reference agent **0.8795** on this setup (§9.4) and
upstream published 88.04. The bf16 on-device stack reproduces the reference PDM
within ~0.07 %.

### 9.1 Eval environment (customize to your machine)

Every external eval asset is addressed through an **environment variable** so
nothing is hard-coded to one machine. The defaults below assume all assets are
staged under a single `DD_DATA_ROOT` (the reference layout — point each var at
wherever you actually put each piece):

```bash
# --- Base dirs -------------------------------------------------------------
export TT_METAL_HOME=/path/to/tt-metal           # <-- EDIT: your tt-metal checkout
export DD_DATA_ROOT=/mnt/diffusion-drive         # <-- EDIT: where eval assets live

# --- Model assets (derived from DD_DATA_ROOT) ------------------------------
export DD_CHECKPOINT_PATH=$DD_DATA_ROOT/weights/diffusiondrive_navsim_88p1_PDMS.pth
export DD_ANCHOR_PATH=$DD_DATA_ROOT/resnet34/kmeans_navsim_traj_20.npy

# --- NavSim devkit + data (standard NavSim env vars) -----------------------
export NAVSIM_DEVKIT_ROOT=$DD_DATA_ROOT/DiffusionDrive    # the NavSim devkit checkout
export NAVSIM_EXP_ROOT=$DD_DATA_ROOT/exp                  # experiment output + metric cache
export OPENSCENE_DATA_ROOT=$NAVSIM_DEVKIT_ROOT/download   # OpenScene sensors + navsim_logs
export NUPLAN_MAPS_ROOT=$DD_DATA_ROOT/dataset/maps        # nuPlan maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

The eval runs in the `navsim` conda env (Python 3.10 — the navsim stack plus
`ttnn`); activate it with `conda activate navsim`.

> **Not env-var-driven — you must customize these before the PDM scoring step.**
> A few things are intrinsically machine-specific and cannot be parameterized away:
>
> - **The `navsim` conda env** — the one-time `pip install --no-deps …` step (see
>   the inproc README) targets *this env's* `pip`; run it inside the activated env
>   (`conda activate navsim`) rather than hard-coding a
>   `/…/miniconda3/envs/navsim/bin/pip` path.
> - **`DD_DATA_ROOT` sub-layout** — `weights/`, `resnet34/`, `DiffusionDrive/`,
>   `exp/`, `dataset/maps/` are the reference sub-paths. If you staged assets
>   differently, set `DD_CHECKPOINT_PATH`, `DD_ANCHOR_PATH`, `NAVSIM_DEVKIT_ROOT`,
>   `NUPLAN_MAPS_ROOT`, etc. **individually** instead of relying on `DD_DATA_ROOT`.
> - **The NavSim devkit itself is required and not relocatable once installed** —
>   `navsim` is an *editable* install pointing at `$NAVSIM_DEVKIT_ROOT`.

These same vars are read directly by the code: the scripts default
`--checkpoint`/`--anchors` to `$DD_CHECKPOINT_PATH`/`$DD_ANCHOR_PATH`, the agent
YAMLs interpolate them via `${oc.env:…}`, and the PCC gates fall back to
`$DD_DATA_ROOT/weights/…`.

### 9.2 Assets & one-time provisioning

**Assets** (everything staged under `$DD_DATA_ROOT`):

| Env var / file | Reference path | What it is |
|---|---|---|
| `NAVSIM_DEVKIT_ROOT` | `$DD_DATA_ROOT/DiffusionDrive` | NavSim devkit (`run_pdm_score.py`, `navsim` pkg, hydra configs) |
| `OPENSCENE_DATA_ROOT` | `$NAVSIM_DEVKIT_ROOT/download` | OpenScene sensor blobs + `navsim_logs` |
| `NUPLAN_MAPS_ROOT` | `$DD_DATA_ROOT/dataset/maps` | nuPlan maps |
| `NAVSIM_EXP_ROOT` | `$DD_DATA_ROOT/exp` | experiment output **and the metric cache** (§9.3) |
| `DD_CHECKPOINT_PATH` | `$DD_DATA_ROOT/weights/diffusiondrive_navsim_88p1_PDMS.pth` | trained 88.x checkpoint (≈700 MB) |
| `DD_ANCHOR_PATH` | `$DD_DATA_ROOT/resnet34/kmeans_navsim_traj_20.npy` | K=20 plan anchors (20×8×2) |
| `pytorch_model.bin` | `$DD_DATA_ROOT/resnet34/pytorch_model.bin` | **required** ImageNet ResNet-34 init (`timm/resnet34.a1_in1k`) — **no env var**: its path is the `bkb_path` you patch in `transfuser_config.py` (below) |

For the *PCC tests* (§6) the checkpoint + anchors can instead come from
`scripts/prepare_assets.py` (into the repo `data/` dir). For the *PDM eval*, stage
everything under `$DD_DATA_ROOT` once — this is NavSim-devkit-side setup the model
repo does not automate:

```bash
# 1. NavSim devkit (python310 branch) + its conda env
cd $DD_DATA_ROOT && git clone -b python310 https://github.com/ayewo/DiffusionDrive
cd DiffusionDrive && conda env create --name navsim -f environment.yml
conda activate navsim && pip install -r requirements.txt && pip install -e .

# 2. OpenScene navtest split (sensors + logs), then strip the test_ prefix navtest expects
cd $OPENSCENE_DATA_ROOT && ./download_test.sh            # large; ~1.5 h
ln -s $OPENSCENE_DATA_ROOT/test_navsim_logs  $OPENSCENE_DATA_ROOT/navsim_logs
ln -s $OPENSCENE_DATA_ROOT/test_sensor_blobs $OPENSCENE_DATA_ROOT/sensor_blobs

# 3. nuPlan maps
cd $OPENSCENE_DATA_ROOT
curl -LO https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
unzip nuplan-maps-v1.1.zip && mv nuplan-maps-v1.0 $NUPLAN_MAPS_ROOT

# 4. Checkpoint, anchors, resnet34 backbone
mkdir -p $DD_DATA_ROOT/weights $DD_DATA_ROOT/resnet34
huggingface-cli download hustvl/DiffusionDrive --local-dir $DD_DATA_ROOT/weights
mv $DD_DATA_ROOT/weights/diffusiondrive_navsim_88p1_PDMS{,.pth}
cd $DD_DATA_ROOT/resnet34
curl -LO https://huggingface.co/timm/resnet34.a1_in1k/resolve/main/pytorch_model.bin
curl -LO https://github.com/hustvl/DiffusionDrive/releases/download/DiffusionDrive_88p1_PDMS_Eval_file/kmeans_navsim_traj_20.npy
```

> **Two hard-coded paths to patch — not env-var-driven.** The devkit's
> `navsim/agents/diffusiondrive/transfuser_config.py` hard-codes the resnet34
> backbone (`bkb_path`) and plan-anchor (`plan_anchor_path`) to the upstream
> author's home dir. Repoint them at your `$DD_DATA_ROOT/resnet34/` — needed only
> for the **PyTorch reference** agent (`agent=diffusiondrive_agent`); the TTNN
> agent reads `$DD_ANCHOR_PATH` from its YAML:
> ```bash
> cfg=$NAVSIM_DEVKIT_ROOT/navsim/agents/diffusiondrive/transfuser_config.py
> sed -i "s|bkb_path: str = .*|bkb_path: str = \"$DD_DATA_ROOT/resnet34/pytorch_model.bin\"|" $cfg
> sed -i "s|plan_anchor_path: str = .*|plan_anchor_path: str = \"$DD_DATA_ROOT/resnet34/kmeans_navsim_traj_20.npy\"|" $cfg
> ```

### 9.3 One-time: build the PDM metric cache (required before evaluating)

Before the eval can run, you must build the PDM metric cache once.
`run_pdm_score.py` does **not** compute it on the fly — it expects a precomputed
cache at `$NAVSIM_EXP_ROOT/metric_cache` (~3.1 GB for `navtest`), which does not
exist on a fresh machine:

```bash
conda activate navsim
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    worker=ray_distributed
# builds $NAVSIM_EXP_ROOT/metric_cache/ (the config default cache.cache_path) — one-time,
# reused by every eval run. ~1.3 h wall on a 4-core host; caches all 12146 navtest scenes.
```

It reads `NAVSIM_EXP_ROOT`, `OPENSCENE_DATA_ROOT`, `NUPLAN_MAPS_ROOT`, and
`NUPLAN_MAP_VERSION` from the §9.1 block. Verify before evaluating:

```bash
du -shL $NAVSIM_EXP_ROOT/metric_cache    # expect ~3.1 GB
```

### 9.4 Run the eval (default: in-process agent)

Full recipe — device-arbitration `worker=…` choices, trace capture (`DD_TRACE`),
and the one-time "make `ttnn` importable in the navsim env" steps — is in
[`scripts/navsim_inproc/README.md`](scripts/navsim_inproc/README.md). Short form:

```bash
conda activate navsim
BR=$TT_METAL_HOME/models/demos/diffusion_drive/scripts/navsim_inproc

# ttnn must import in the navsim env: one-time deps + the inner-package parent on
# PYTHONPATH (just $TT_METAL_HOME resolves to an empty namespace pkg — see sub-README).
export TTNN_PP=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools
export PYTHONPATH=$BR:$TTNN_PP:$NAVSIM_DEVKIT_ROOT

# install the agent's hydra config into the devkit (one-time)
cp $BR/diffusiondrive_ttnn_inproc_agent.yaml \
   $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/agent/

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=diffusiondrive_ttnn_inproc_agent \
    worker=single_machine_thread_pool \
    experiment_name=diffusiondrive_ttnn_inproc_eval
```

The `Final average score of valid results: …` line is the PDM score (expect
~0.8789). Cap a smoke run with `train_test_split.scene_filter.max_scenes=5`, and
validate parity first with `scripts/navsim_inproc/check_parity.py` (§0 of the
sub-README). The agent reads the checkpoint/anchors from the YAML, which
interpolate `$DD_CHECKPOINT_PATH`/`$DD_ANCHOR_PATH`.

**Reproduce the PyTorch reference baseline (optional).** For the comparison
number, run the upstream PyTorch agent instead — it needs the resnet34 backbone +
the `transfuser_config.py` patches from §9.2:

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=diffusiondrive_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$DD_CHECKPOINT_PATH \
    experiment_name=diffusiondrive_agent_eval
# this setup: PDM 0.8795 over 12146/12146, ~48 min wall
```
