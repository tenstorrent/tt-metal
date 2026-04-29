# VGGT on Tenstorrent Blackhole p150a

VGGT (Visual Geometry Grounded Deep Structure From Motion) is a transformer model from Meta that predicts dense 3D structure — depth maps, 3D world points, and camera poses — from one or more unposed images in a single forward pass.

This directory contains a ttnn port of [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) targeting the Tenstorrent Blackhole p150a chip.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         VGGT-1B                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input: (B, S, 3, 518, 518)  — B scenes, S views each             │
│                                                                    │
│  ┌─────────────────────────────────────────────┐                  │
│  │  DINOv2 ViT-L/14  (patch=14, 24 blocks)     │                  │
│  │  518×518 → 37×37 = 1369 patches + 5 regs   │                  │
│  │  = 1374 tokens/frame, hidden_dim=1024        │                  │
│  └────────────────────┬────────────────────────┘                  │
│                       │  tokens (B*S, 1374, 1024)                 │
│  ┌────────────────────▼────────────────────────┐                  │
│  │  Aggregator  (frame_blocks × 24 +           │                  │
│  │               global_blocks × 24)           │                  │
│  │  Frame attn: each view attends within self  │                  │
│  │  Global attn: all S×1374 tokens attend      │                  │
│  │  jointly; canonical-S padding for BF0       │                  │
│  └────────┬──────────────────────┬─────────────┘                  │
│           │                      │                                 │
│  ┌────────▼──────────┐  ┌────────▼──────────┐                     │
│  │  DPT Head         │  │  Camera Head      │                     │
│  │  4 DPT layers     │  │  16-layer trunk   │                     │
│  │  + scratch convs  │  │  pose branch      │                     │
│  │  → depth,         │  │  → pose_enc       │                     │
│  │    world_points   │  │                   │                     │
│  └───────────────────┘  └───────────────────┘                     │
│                                                                    │
│  Output: depth, depth_conf, world_points, world_points_conf,       │
│          pose_enc  — all (B, S, …) tensors                         │
└────────────────────────────────────────────────────────────────────┘
```

## On-device port summary

| Component | Status | Notes |
|-----------|--------|-------|
| DINOv2 transformer blocks (24) | ✅ On device | `_tt_can_pass=True` → residual stays on device |
| Aggregator frame blocks (24) | ✅ On device | `_TTPassed` eliminates PCIe round-trips at S=1 |
| Aggregator global blocks (24) | ✅ On device | BF0: canonical-S padding for variable S |
| 2D RoPE | ✅ On device | cos/sin tables precomputed at install |
| Camera head trunk blocks | ✅ On device | |
| DPT scratch output_conv2 | ✅ On device | 3×3 + ReLU + 1×1 at 518×518, HiFi4 |
| DPT prelude (norm+projects+resize) | ⚠️ CPU | Device port implemented (`VGGT_TT_PRELUDE=1`), no throughput gain at S=1 |
| DPT scratch refinenets | ❌ CPU | 120 ops × round-trip overhead > compute gain |
| custom_interpolate bilinear | ❌ CPU | ~13 ms, low priority |

## Precision

- Weights and matmul inputs: **bfloat16**
- Residual accumulator (Block fp32 via `dtype=float32` on `proj`/`fc2` outputs)
- Attention scores + softmax + context: **fp32** (HiFi4 + `dtype=float32`)
  - bf16 softmax over 1374-token rows collapsed `world_points_conf` PCC below 0.99
  - FlashAttention-2 / SDPA also dropped (`world_points_conf` PCC = 0.89 due to bf16 softmax internally)

## Performance

Measured on a single Blackhole p150a chip, B=1, 518×518 input:

| S (views) | Latency | Throughput | PCC (min) | Notes |
|-----------|---------|------------|-----------|-------|
| 1 | 1294 ms | 0.773 fps | 0.9957 | 3.9× vs CPU baseline (5037 ms) |
| 2 | ~1540 ms | ~1.30 fps | 0.9957 | |
| 3 | ~1480 ms | ~2.03 fps | 0.9957 | BF0 canonical-S padding |
| 4 | ~1560 ms | ~2.56 fps | 0.9957 | |

CPU torch baseline: 5037 ms / frame, 0.199 fps.

## CO3Dv2 evaluation results

12 scenes, 6 categories (apple, bottle, chair, laptop, hydrant, teddybear):

| S | Pairs | RRA@5° | RRA@15° | RTA@5° | RTA@15° | AUC@30° | Δ AUC vs ref |
|---|-------|--------|---------|--------|---------|---------|-------------|
| 1 | 12 | 100 % | 100 % | 100 % | 100 % | 96.8 | — |
| 3 | 36 | 100 % | 100 % | 100 % | 100 % | 96.9 | −0.1 |
| 4 | 72 | 100 % | 100 % | 100 % | 100 % | 96.9 | 0.0 |

At S=4 the port **ties the reference exactly** (Δ AUC = 0.0, all PCC channels ≥ 0.9971).

## Directory structure

```
vggt/
├── reference/
│   ├── __init__.py
│   └── torch_vggt.py          # PyTorch reference loader (facebook/VGGT-1B)
├── tt/
│   ├── __init__.py
│   └── ttnn_vggt.py           # Full ttnn port (install + forward)
├── tests/
│   ├── __init__.py
│   ├── test_vggt.py           # pytest PCC + perf tests; also runnable as CLI
│   └── eval_co3d.py           # CO3Dv2 camera-pose accuracy evaluation
└── README.md
```

## Quick start

### 1. Environment setup

```bash
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=blackhole
source $TT_METAL_HOME/python_env/bin/activate
```

### 2. Weights and upstream source

```bash
# Clone the upstream VGGT source (needed for model definition)
git clone https://github.com/facebookresearch/vggt /path/to/vggt_ref

# Download weights from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/VGGT-1B')"
```

Set paths via environment variables (or accept the defaults if running on the Tenstorrent dev machine):

```bash
export VGGT_REF_PATH=/path/to/vggt_ref
export VGGT_WEIGHTS_PATH=/path/to/model.safetensors
```

### 3. Run PCC accuracy tests

```bash
# S=1 and S=2 PCC tests (threshold 0.99)
pytest $TT_METAL_HOME/models/experimental/vggt/tests/test_vggt.py -v

# Specific seq length
pytest $TT_METAL_HOME/models/experimental/vggt/tests/test_vggt.py \
    -v -k "seq-1"
```

### 4. Run performance benchmark

```bash
pytest $TT_METAL_HOME/models/experimental/vggt/tests/test_vggt.py::test_perf_s1 -v -s
```

### 5. CLI benchmark (direct)

```bash
python $TT_METAL_HOME/models/experimental/vggt/tests/test_vggt.py \
    --seq 1 --runs 3 --device-id 0
```

### 6. CO3Dv2 evaluation

```bash
python $TT_METAL_HOME/models/experimental/vggt/tests/eval_co3d.py \
    --co3d-root /path/to/co3d_data \
    --categories apple,bottle,chair,laptop,hydrant,teddybear \
    --num-views 4 --device-id 0
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_METAL_HOME` | auto-detected | Root of tt-metal checkout |
| `TT_DEVICE_ID` | `0` | Tenstorrent chip index |
| `VGGT_REF_PATH` | `/home/ttuser/experiments/vggt/vggt_ref` | facebook/VGGT upstream source |
| `VGGT_WEIGHTS_PATH` | HF cache path | `model.safetensors` from facebook/VGGT-1B |
| `VGGT_S_CANON` | matches `--seq` | Canonical S for BF0 padding |
| `VGGT_TT_PRELUDE` | `0` | Enable device-side DPT prelude (`1` to enable) |
| `VGGT_TT_SCRATCH` | `1` | Enable device-side DPT output_conv2 |

## Model specifications

| Component | Details |
|-----------|---------|
| Image encoder | DINOv2 ViT-L/14 (24 transformer blocks, 1024 hidden dim) |
| Aggregator | 24 frame blocks + 24 global blocks (1024 hidden dim) |
| Camera head | 16-layer trunk + pose branch |
| DPT head | 4 DPT layers + scratch refinenet + output_conv2 |
| Input size | 518 × 518 (patch size 14 → 37×37 grid) |
| Tokens/frame | 1374 (1369 patches + 5 register tokens) |
| Parameters | ~1B |

## License

SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
