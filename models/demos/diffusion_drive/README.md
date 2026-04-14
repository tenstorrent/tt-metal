# DiffusionDrive Autonomous Driving Planner (TTNN)

## Platforms
- Wormhole (`n300`)

## Overview

DiffusionDrive (CVPR 2025) is a diffusion-based trajectory planner for autonomous
driving.  This bring-up replaces its PyTorch ops with native TTNN kernels one stage
at a time, validating PCC ≥ 0.99 at each step.

**Architecture summary**

| Component | Description |
|---|---|
| Image encoder | ResNet-34 (timm, `features_only`) — 4 stages, BN-folded |
| LiDAR encoder | ResNet-34 (timm, `features_only`) — 4 stages, BN-folded |
| GPT fusion | 4-scale cross-modal transformer (AdaptiveAvgPool2d + 2-layer GPT) |
| FPN | 3-level top-down (`c5_conv` 1×1 + 2× `up_conv` 3×3, bilinear upsample) |
| Perception decoder | 3-layer TransformerDecoder (d=256, 8 heads, 31 queries × 65 key-val tokens) |
| Trajectory head | DDIM 2-step denoiser, K=20 anchors, T=8 waypoints |
| Outputs | `trajectory` (B×8×3), `scores` (B×20) |

Model checkpoint: `hustvl/DiffusionDrive` on HuggingFace (60 M parameters).

## Directory Layout

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
│   ├── pcc/
│   │   ├── test_pcc_bn_fold.py         # BN-fold accuracy (6 tests)
│   │   ├── test_pcc_resnet_block.py    # BasicBlock PCC (3 tests)
│   │   ├── test_pcc_backbone.py        # Stage 2 backbone PCC (2 tests)
│   │   ├── test_pcc_fpn.py             # Stage 3 FPN PCC (1 test)
│   │   ├── test_pcc_full_model.py      # Stage 1 full-model PCC (2 tests)
│   │   ├── test_pcc_stage2.py          # Stage 2 full-model PCC (2 tests)
│   │   └── test_pcc_stage3.py          # Stage 3 full-model PCC (2 tests)
│   └── sanity/
│       └── test_no_nan_inf.py          # no NaN/Inf, score/trajectory range (3 tests)
└── tt/
    ├── config.py                # ModelConfig dataclass
    ├── ttnn_resnet34.py         # TtnnBasicBlock (BN-fold + TTNN conv2d)
    ├── ttnn_backbone.py         # TtnnTransfuserBackbone (Stage 2)
    ├── ttnn_fpn.py              # TtnnFPN (Stage 3)
    └── ttnn_diffusion_drive.py  # TtnnDiffusionDriveModel (build_stage2/3)
```

## Setup

```bash
# 1. Create and activate the project virtual environment
./create_venv.sh
source python_env/bin/activate
export PYTHONPATH=/root/tt/tt-metal

# 2. Verify device access
python3 -m ttnn.examples.usage.run_op_on_device   # should print a bfloat16 tensor
tt-smi                                             # device info
```

### Environment fixes (Wormhole N300s — one-time, machine-local)

**a) sfpi runtime alias** — JIT compilation of `ttnn.conv2d` fails with
`` 'sfpi::sFloat16b' is not a member of 'sfpi' `` unless the following aliases
are added inside the `sfpi` namespace in
`runtime/sfpi/include/sfpi_fp16.h` (the `runtime/` directory is gitignored):

```cpp
using sFloat16a = s2vFloat16a;
using sFloat16b = s2vFloat16b;
```

**b) l1_small_size** — the test `conftest.py` already opens the device with
`l1_small_size=32768`.  If you open a device manually, pass the same argument:

```python
device = ttnn.open_device(device_id=0, l1_small_size=32768)
```

## Assets

Download the NavSim checkpoint and extract the K=20 trajectory anchor clusters:

```bash
python models/demos/diffusion_drive/scripts/prepare_assets.py
```

Generated files:

| File | Description |
|---|---|
| `data/diffusiondrive_navsim.pth` | Full model checkpoint (hustvl/DiffusionDrive) |
| `data/kmeans_navsim_traj_20.npy` | K-means anchor array, shape (20, 8, 2) |

Tests that require these files skip automatically if they are absent.

## Running Tests

```bash
source python_env/bin/activate
export PYTHONPATH=/root/tt/tt-metal

# All PCC tests (require attached Wormhole device)
python -m pytest models/demos/diffusion_drive/tests/pcc/ -v

# Sanity tests only (no device needed)
python -m pytest models/demos/diffusion_drive/tests/sanity/ -v

# Full suite
python -m pytest models/demos/diffusion_drive/tests/ -v
```

Run a single test file:

```bash
# Stage 3 FPN conv accuracy
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_fpn.py -v

# Stage 3 full-model trajectory and scores
python -m pytest models/demos/diffusion_drive/tests/pcc/test_pcc_stage3.py -v
```

## Bring-Up Stage Status

| Stage | Scope | Tests | Commit |
|---|---|---|---|
| Stage 0 | Architecture audit, reference model confirmed | n/a | — |
| Stage 1 | PyTorch reference wrapper + BN-fold + BasicBlock PCC | 14/14 | `857671c0aa` |
| Stage 2 | All 32 ResNet-34 BasicBlock stages on TTNN (`ttnn.conv2d`, BN-fold) | 15/15 | `a72716b165` |
| Stage 3 | 3-level FPN conv2d on TTNN (`TtnnFPN`; bilinear upsample stays PyTorch) | 18/18 | `edd70f9e9f` |

Current total: **21 tests pass** (18 PCC + 3 sanity).

### What runs on TTNN (Stage 3)

- All 32 ResNet-34 BasicBlock layers (image_encoder + lidar_encoder × 4 stages)
- FPN `c5_conv` (1×1, 512→64), `up_conv5` (3×3, 64→64), `up_conv4` (3×3, 64→64)

### What stays in PyTorch

- Stem: `conv1 + bn1 + act1 + MaxPool2d`
- GPT cross-modal fusion (AdaptiveAvgPool2d + 2-layer transformer + F.interpolate + residual add)
- FPN bilinear upsampling (`F.interpolate` — `ttnn.upsample` does not support bilinear)
- Perception TransformerDecoder (3 layers, d=256, 8 heads)
- TrajectoryHead (DDIM 2-step, `F.grid_sample` deformable attention)

## Usage

```python
import ttnn
import torch
from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel
from models.demos.diffusion_drive.tt.config import ModelConfig
from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, DiffusionDriveModel

# Open device
device = ttnn.open_device(device_id=0, l1_small_size=32768)

# Load reference model (requires prepare_assets.py to have run)
cfg = DiffusionDriveConfig(
    plan_anchor_path="models/demos/diffusion_drive/data/kmeans_navsim_traj_20.npy"
)
ref_model = DiffusionDriveModel(cfg).eval()

# Build Stage 3 TTNN model
model_config = ModelConfig()
ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
ttnn_model.build_stage2(device).build_stage3(device)

# Run inference
features = {
    "camera_feature": torch.randn(1, 3, 256, 1024),
    "lidar_feature":  torch.zeros(1, 1, 256, 256),
    "status_feature": torch.zeros(1, 8),
}
out = ttnn_model(features)
# out["trajectory"]: (1, 8, 3)  — best trajectory (x, y, heading) per timestep
# out["scores"]:     (1, 20)    — anchor mode log-scores

ttnn.close_device(device)
```

## Notes

- **lidar_resolution must be ≥ 256×256** for full-model tests.  The perception
  decoder's positional embedding expects exactly 65 key-val tokens (8×8 BEV + 1
  status), which requires the LiDAR encoder's `layer4` to produce an 8×8 spatial
  map.  A 64×64 LiDAR input yields 2×2=4 tokens and crashes with a size mismatch.
  Backbone-only PCC tests may use 64×64.

- **DDIM noise seeding** — `TrajectoryHead._forward_test` calls `torch.randn`
  for DDIM noise at each forward pass.  PCC tests pin `torch.manual_seed(1234)`
  before each of the two forward calls (reference and TTNN) to keep noise
  identical.

- **Conv2dConfig** — the compiled binary only accepts `weights_dtype`, not `dtype`.
  Do not pass `dtype=ttnn.bfloat16` to `Conv2dConfig()`.
