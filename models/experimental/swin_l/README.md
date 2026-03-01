# Swin-L Backbone (TTNN)

Standalone, reusable TTNN implementation of the **Swin Transformer Large** backbone.
Produces 4 multi-scale feature maps (C2–C5), ready for downstream detection/segmentation heads.

## Architecture

| Stage | Blocks | Channels | Heads | Window | Output Scale |
|-------|--------|----------|-------|--------|-------------|
| 0     | 2      | 192      | 6     | 12     | H/4 × W/4  |
| 1     | 2      | 384      | 12    | 12     | H/8 × W/8  |
| 2     | 18     | 768      | 24    | 12     | H/16 × W/16|
| 3     | 2      | 1536     | 48    | 12     | H/32 × W/32|

## Quick Start

### 1. Prerequisites

```bash
source python_env/bin/activate
pip install --user openmim
mim install --user mmdet
export PYTHONPATH=$TT_METAL_HOME:$HOME/.local/lib/python3.10/site-packages
```

### 2. Checkpoint

Any mmdet checkpoint containing a Swin-L backbone works.
For example, DINO-5scale:

```bash
mim download mmdet \
    --config dino-5scale_swin-l_8xb2-36e_coco \
    --dest models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l
```

Or set env vars to use a custom checkpoint:

```bash
export SWIN_L_CONFIG=/path/to/your/config.py
export SWIN_L_CKPT=/path/to/your/checkpoint.pth
```

### 3. Usage

```python
from models.experimental.swin_l.tt import TtSwinLBackbone, load_backbone_weights, compute_attn_masks

# Load weights from any mmdet checkpoint with Swin-L backbone
params = load_backbone_weights(checkpoint_path, device)
attn_masks = compute_attn_masks(input_h, input_w, patch_size=4, window_size=12, device=device)

model = TtSwinLBackbone(device, params, attn_masks=attn_masks)
features = model(input_nchw)  # Returns list of 4 NCHW feature maps
```

### 4. Run PCC Tests

```bash
export PYTHONPATH=$TT_METAL_HOME:$HOME/.local/lib/python3.10/site-packages

# All submodule tests
pytest models/experimental/swin_l/tests/pcc/ -v

# Individual tests
pytest models/experimental/swin_l/tests/pcc/test_ttnn_backbone.py -v
pytest models/experimental/swin_l/tests/pcc/test_ttnn_swin_attention.py -v
pytest models/experimental/swin_l/tests/pcc/test_ttnn_swin_mlp.py -v
pytest models/experimental/swin_l/tests/pcc/test_ttnn_swin_block.py -v
pytest models/experimental/swin_l/tests/pcc/test_ttnn_swin_patch_merge.py -v
```

### 5. Run Performance Tests

Requires a DINO-5scale Swin-L checkpoint (see §2). Runs on Wormhole B0 bare metal.

```bash
# E2E perf (trace + 2CQ)
pytest models/experimental/swin_l/tests/perf/test_e2e_perf_swin_l.py -v -s

# Device perf
pytest models/experimental/swin_l/tests/perf/test_swin_l_device_perf.py -v -s
```

## PCC Results (bfloat16, DRAM)

### Backbone E2E

| Stage | Channels | PCC   |
|-------|----------|-------|
| 0     | 192      | 0.997 |
| 1     | 384      | 0.997 |
| 2     | 768      | 0.984 |
| 3     | 1536     | 0.994 |

### Submodules

| Module       | Test Case         | PCC   |
|--------------|-------------------|-------|
| Attention    | no shift          | 0.998 |
| Attention    | with shift        | 0.999 |
| MLP (FFN)    | stage 0 block 0   | 0.999 |
| Block        | s0 b0 (no shift)  | 0.996 |
| Block        | s0 b1 (shift)     | 1.000 |
| Block        | s2 b0 (deep)      | 0.999 |
| PatchMerge   | stage 0           | 1.000 |
| PatchMerge   | stage 1           | 1.000 |

## Performance Results (Wormhole B0, 800×1333, batch=1)

| Test                    | Throughput (FPS) |
|-------------------------|------------------|
| E2E trace + 2CQ         | ~1.9             |
| Device perf             | ~2.04            |

## File Structure

```
models/experimental/swin_l/
├── common.py                          # Swin-L architecture constants
├── tt/
│   ├── __init__.py                    # Public API exports
│   ├── tt_backbone.py                 # TtSwinLBackbone (full model)
│   ├── tt_swin_attention.py           # TtSwinAttention
│   ├── tt_swin_mlp.py                 # TtSwinMLP
│   ├── tt_swin_block.py               # TtSwinBlock
│   ├── tt_swin_patch_merge.py         # TtSwinPatchMerge
│   └── model_preprocessing.py         # Weight loading from mmdet checkpoints
├── reference/
│   └── swin_l_reference.py            # PyTorch reference (mmdet wrapper)
├── tests/pcc/
│   ├── conftest.py                    # Shared fixtures
│   ├── test_ttnn_backbone.py          # E2E backbone PCC test
│   ├── test_ttnn_swin_attention.py    # Attention PCC test
│   ├── test_ttnn_swin_mlp.py          # MLP PCC test
│   ├── test_ttnn_swin_block.py        # Block PCC test
│   └── test_ttnn_swin_patch_merge.py  # PatchMerge PCC test
├── tests/perf/
│   ├── test_swin_l_device_perf.py     # Device perf
│   └── test_e2e_perf_swin_l.py        # E2E perf (trace + 2CQ)
└── README.md
```

## Importing in Other Models

```python
# In your model (e.g., DINO, ATSS, etc.)
from models.experimental.swin_l.tt import TtSwinLBackbone, load_backbone_weights, compute_attn_masks
```
