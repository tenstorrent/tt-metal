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
9. [Future Work](#9-future-work)

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
  └─► conv1+bn1+maxpool (PyTorch stem)
       └─► layer1…layer4 (TTNN — 16 BasicBlocks) ──────────────────────┐
                                                                         │
LiDAR (B,1,256,256)                                                      │
  └─► conv1+bn1+maxpool (PyTorch stem)                         GPT fusion
       └─► layer1…layer4 (TTNN — 16 BasicBlocks) ──────────── ×4 scales │
                                                   (PyTorch ← AdaptiveAvgPool +
                                                    2-layer GPT + F.interpolate)
                                                                         │
                                                              lidar_feats (B,512,8,8)
                                                                         │
                                                    ┌────────────────────┤
                                                    │         bev_feature│
                                          FPN _top_down                  │
                                  c5_conv 1×1  (TTNN)                   │
                                  upsample 2×  (PyTorch F.interpolate)  │
                                  up_conv5 3×3 (TTNN)                   │
                                  upsample to 64×64 (PyTorch)           │
                                  up_conv4 3×3 (TTNN)                   │
                                       │                                 │
                                bev_upscale (B,64,64,64)                │
                                                                         │
                           _bev_downscale (PyTorch 1×1 Conv2d)         │
                           _status_encoding (PyTorch Linear)            │
                           _keyval_embedding (PyTorch add)              │
                           F.interpolate + cat bev_upscale              │
                           bev_proj: Linear(320,256)+ReLU+LN (PyTorch) │
                                       │                                 │
                           _tf_decoder (PyTorch TransformerDecoder)    │
                                       │                                 │
                           TrajectoryHead DDIM ◄──────────── bev_feature┘
                                       │
                             trajectory (B,8,3)  scores (B,20)
```

### TTNN vs PyTorch per submodule

| Submodule | Stage 3 execution | Reason for placement |
|---|---|---|
| ResNet-34 stems (conv1+bn1+maxpool) | PyTorch | Small one-time cost; deferred |
| ResNet-34 BasicBlocks ×32 | **TTNN** | High FLOP density; BN-fold enables weight-only conv |
| GPT cross-modal fusion ×4 | PyTorch | `AdaptiveAvgPool2d` + `F.interpolate` interleaved with attention; mixed-mode migration complex |
| FPN conv layers ×3 | **TTNN** | Plain `nn.Conv2d`, no BN; direct weight cast |
| FPN bilinear upsample ×2 | PyTorch | `ttnn.upsample` is nearest-only; bilinear diverges at these scales |
| `_bev_downscale` 1×1 conv | PyTorch | Deferred to Stage 4 |
| Perception TransformerDecoder | PyTorch | Deferred to Stage 4 |
| TrajectoryHead DDIM + grid_sample | PyTorch | `F.grid_sample` has no TTNN equivalent |

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

### 3.4 Why FPN bilinear upsample stays in PyTorch

The FPN needs two bilinear upsamples (8×8→16×16 and 16×16→64×64).
`ttnn.upsample` currently supports nearest-neighbour only.  Using nearest instead
of bilinear on these small feature maps produces non-trivial pixel differences that
propagate through the downstream transformer and drop full-model PCC below 0.99.
Until bilinear is supported in TTNN, `F.interpolate(mode="bilinear")` is used
in-place.

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
| 2 | All 32 ResNet-34 BasicBlock conv layers on TTNN | +32 | 15/15 | `a72716b165` |
| 3 | FPN 3 conv layers on TTNN (`TtnnFPN`) | +3 | 18/18 | `edd70f9e9f` |

**Current total: 21 tests pass** (18 PCC + 3 sanity).  35 `ttnn.conv2d` ops run on
device per forward pass.

### TTNN ops on-device at Stage 3

```
image_encoder.layer1  — 6 BasicBlock conv2d ops  (64-ch, 64×256 spatial)
image_encoder.layer2  — 8 BasicBlock conv2d ops  (128-ch, 32×128 spatial)
image_encoder.layer3  — 12 BasicBlock conv2d ops (256-ch, 16×64 spatial)
image_encoder.layer4  — 6 BasicBlock conv2d ops  (512-ch, 8×32 spatial)

lidar_encoder.layer1  — 6 BasicBlock conv2d ops  (64-ch, 64×64 spatial)
lidar_encoder.layer2  — 8 BasicBlock conv2d ops  (128-ch, 32×32 spatial)
lidar_encoder.layer3  — 12 BasicBlock conv2d ops (256-ch, 16×16 spatial)
lidar_encoder.layer4  — 6 BasicBlock conv2d ops  (512-ch, 8×8 spatial)

FPN c5_conv           — 1×1 conv2d  (512→64, 8×8 spatial)
FPN up_conv5          — 3×3 conv2d  (64→64, 16×16 spatial)
FPN up_conv4          — 3×3 conv2d  (64→64, 64×64 spatial)
                                                    ─────────
                                                    35 ops total
```

---

## 5. Performance

Measured on Wormhole N300s, batch=1, full production resolution
(camera 256×1024, LiDAR 256×256), `latent=True`.
Each figure is the minimum over 5 runs after 2 warm-up calls.

### Latency

| Stage | Min latency | vs Stage 1 | On-device ops |
|---|---|---|---|
| Stage 1 — pure PyTorch | 652 ms | baseline | 0 |
| Stage 2 — TTNN backbone | 540 ms | **−17%** | 32 |
| Stage 3 — TTNN backbone + FPN | 486 ms | **−25%** | 35 |

The remaining ~486 ms is dominated by the GPT cross-modal fusion (4 × 2-layer
transformer + pooling) and the TrajectoryHead DDIM denoiser, both still in PyTorch.
These are the primary targets for Stages 4 and 5.

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
# ... build ttnn_model (stage 2 or 3) ...

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
./create_venv.sh
source python_env/bin/activate
export PYTHONPATH=/root/tt/tt-metal

# 2. Verify device access
python3 -m ttnn.examples.usage.run_op_on_device   # should print a bfloat16 tensor
tt-smi                                             # device info
```

### One-time environment fixes (Wormhole N300s)

These are machine-local patches that survive `git pull` but are lost if
`runtime/` is deleted and recreated.

**a) sfpi runtime alias**

`ttnn.conv2d` triggers JIT compilation of `pack_untilize.cpp`.  The tt-llk
headers at commit `8dab3a5982` renamed `s2vFloat16b` → `sFloat16b`, but the
bundled `runtime/sfpi/include/sfpi_fp16.h` only defines the old name.
Add these two aliases inside the `sfpi` namespace:

```cpp
// runtime/sfpi/include/sfpi_fp16.h  — inside namespace sfpi { ... }
using sFloat16a = s2vFloat16a;
using sFloat16b = s2vFloat16b;
```

Without this fix the first `ttnn.conv2d` call fails with:
```
error: 'sfpi::sFloat16b' is not a member of 'sfpi'
```

**b) l1_small_size**

Open all devices used with conv2d with `l1_small_size=32768` (see §3.3).
The test `conftest.py` already does this.

**c) Conv2dConfig keyword**

The compiled binary's `Conv2dConfig.__init__` accepts only `weights_dtype`,
not `dtype`.  Do not pass `dtype=ttnn.bfloat16`.

### Assets

Download the NavSim checkpoint and extract the K=20 anchor clusters:

```bash
python models/demos/diffusion_drive/scripts/prepare_assets.py
```

| File | Description |
|---|---|
| `data/diffusiondrive_navsim.pth` | Full model checkpoint (`hustvl/DiffusionDrive`) |
| `data/kmeans_navsim_traj_20.npy` | K-means anchor array, shape (20, 8, 2) |

Tests that require the anchor file skip automatically when it is absent.

---

## 7. Running Tests

```bash
source python_env/bin/activate
export PYTHONPATH=/root/tt/tt-metal

# Full suite (18 PCC + 3 sanity = 21 tests)
python -m pytest models/demos/diffusion_drive/tests/ -v

# PCC tests only (require attached Wormhole device)
python -m pytest models/demos/diffusion_drive/tests/pcc/ -v

# Sanity tests (no device needed)
python -m pytest models/demos/diffusion_drive/tests/sanity/ -v
```

Individual test files:

```bash
# BN-fold accuracy (6 tests, no device)
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_bn_fold.py -v

# BasicBlock PCC (3 tests)
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_resnet_block.py -v

# Stage 2 backbone outputs (2 tests — bev_upscale, bev_feature)
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_backbone.py -v

# Stage 3 FPN conv output (1 test)
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_fpn.py -v

# Stage 2 full-model trajectory + scores (2 tests)
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_stage2.py -v

# Stage 3 full-model trajectory + scores (2 tests)
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_stage3.py -v
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

# Upgrade to Stage 3 TTNN (chains: build_stage2 then build_stage3)
model_config = ModelConfig()
ttnn_model = (
    TtnnDiffusionDriveModel(ref_model, model_config, device)
    .build_stage2(device)
    .build_stage3(device)
)

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
│   └── prepare_assets.py        # download checkpoint + extract anchors
├── tests/
│   ├── conftest.py              # device fixture (l1_small_size=32768)
│   ├── pcc/                     # accuracy tests (require device)
│   └── sanity/                  # NaN/Inf + range checks (no device)
└── tt/
    ├── config.py                # ModelConfig dataclass
    ├── ttnn_resnet34.py         # TtnnBasicBlock (BN-fold + ttnn.conv2d)
    ├── ttnn_backbone.py         # TtnnTransfuserBackbone (Stage 2)
    ├── ttnn_fpn.py              # TtnnFPN (Stage 3)
    └── ttnn_diffusion_drive.py  # TtnnDiffusionDriveModel (build_stage2/3)
```

---

## 9. Future Work

| Stage | Scope | Key blocker |
|---|---|---|
| 4 | Perception stack: `_bev_downscale` (1×1 conv), `bev_proj` (Linear+ReLU+LN), `_tf_decoder` (3-layer SDPA+FFN) | TTNN SDPA API validation at d=256, 8 heads, 31×65 tokens |
| 5 | TrajectoryHead DDIM denoiser (linear layers and noise schedule) | `F.grid_sample` deformable cross-BEV attention has no TTNN equivalent yet |
| 6 | GPT cross-modal fusion (2-layer transformer ×4 scales) | `AdaptiveAvgPool2d` + `F.interpolate` interleaved with attention; complex mixed-mode migration |
| 7 | Full-stack trace capture | Requires all forward ops on-device to eliminate PCIe round-trips |
