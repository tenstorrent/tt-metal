# ATSS Swin-L DyHead (TTNN)

TTNN implementation of **ATSS** (Adaptive Training Sample Selection) object detection
with **Swin-L** backbone, **FPN** neck, **DyHead** dynamic head, and **ATSS Head**.

## Architecture

```
Input Image [1, 3, 1200, 2000]
  │
  ▼
Swin-L Backbone (TTNN)      → 3 feature maps (stages 1, 2, 3)
  │                            [384, 768, 1536] channels
  ▼
FPN Neck (TTNN)              → 5 feature maps (P3..P7)
  │                            all 256 channels
  ▼
DyHead Neck (Hybrid*)        → 5 refined feature maps
  │                            6 blocks, DCNv2 + scale/spatial/task attention
  ▼
ATSS Head (TTNN)             → per-level: cls (80), reg (4), centerness (1)
  │
  ▼
Post-processing (CPU)        → bboxes, scores, labels
```

*DyHead runs in hybrid mode by default:
- **Spatial attention** (DCNv2) runs on CPU — no native TTNN kernel yet
- **Scale-aware attention** (AvgPool + Conv + hardsigmoid) runs on TTNN
- **Task-aware attention / DyReLU** (AvgPool + FC + hardsigmoid + element-wise) runs on TTNN

Set `hybrid_dyhead=False` in `from_checkpoint()` to run the entire DyHead on CPU.

## Module Structure

```
models/experimental/atss_swin_l_dyhead/
├── common.py                           # ATSS constants + checkpoint auto-download
├── weights/                            # Checkpoint and config (auto-downloaded)
│   ├── atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_*.pth
│   └── atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py
├── reference/                          # PyTorch reference implementations
│   ├── swin_transformer.py            #   Swin-L backbone
│   ├── fpn.py                         #   Feature Pyramid Network
│   ├── dyhead.py                      #   Dynamic Head (DCNv2)
│   ├── atss_head.py                   #   ATSS detection head
│   ├── postprocess.py                 #   Anchor gen, bbox decode, NMS
│   └── model.py                       #   Full model assembly + weight loading
├── tt/                                 # TTNN implementations
│   ├── tt_fpn.py                      #   TTNN FPN
│   ├── tt_dyhead.py                   #   Hybrid DyHead (scale/task on TTNN)
│   ├── tt_atss_head.py                #   TTNN ATSS Head
│   ├── tt_atss_model.py               #   Full TTNN model (hybrid)
│   └── weight_loading.py              #   Weight loading for FPN/DyHead/Head
├── tests/pcc/                          # PCC validation tests
│   ├── conftest.py                    #   Shared fixtures
│   ├── test_reference_model.py        #   Reference model vs mmdet
│   ├── test_ttnn_fpn.py               #   TTNN FPN vs reference
│   ├── test_ttnn_atss_head.py         #   TTNN ATSS Head vs reference
│   └── test_ttnn_e2e.py               #   Full E2E TTNN vs reference
├── demo/                               # Demo scripts
│   ├── demo_inference.py              #   Single-image inference + visualization
│   ├── demo_batch.py                  #   Multi-image batch inference
│   └── demo_perf.py                   #   Performance benchmark (PyTorch vs TTNN)
└── README.md

models/experimental/swin_l/             # Reusable Swin-L backbone (separate module)
├── tt/                                 #   TTNN backbone
│   ├── tt_backbone.py                 #     TtSwinLBackbone
│   ├── tt_swin_attention.py           #     Shifted window attention
│   ├── tt_swin_mlp.py                 #     MLP
│   ├── tt_swin_block.py               #     Transformer block
│   ├── tt_swin_patch_merge.py         #     Patch merging (downsample)
│   └── model_preprocessing.py         #     Backbone weight loading
└── common.py                          #   Swin-L architecture constants
```

## Quick Start

### Prerequisites

```bash
cd $TT_METAL_HOME
source python_env/bin/activate
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages
```

### Checkpoint

The checkpoint is stored inside the model folder at `weights/`. If not present,
it will be auto-downloaded via `mim download mmdet` when you first import `common.py`.

To download manually:
```bash
pip install openmim && mim install mmdet mmengine
mim download mmdet --config atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco \
    --dest models/experimental/atss_swin_l_dyhead/weights/
```

#### Environment variable overrides

| Variable | Description |
|---|---|
| `ATSS_CHECKPOINT` | Path to the `.pth` checkpoint file. Skips auto-download when set. |
| `ATSS_CONFIG` | Path to the mmdet config `.py` file. |

```bash
export ATSS_CHECKPOINT=/data/models/atss_swin-l.pth
export ATSS_CONFIG=/data/models/atss_swin-l-config.py
```

### Run PCC Tests

```bash
# Reference model tests (PyTorch only)
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_reference_model.py -v

# TTNN FPN test
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_fpn.py -v

# TTNN ATSS Head test
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_atss_head.py -v

# Full E2E test
pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_e2e.py -v
```

### Demo

The inference demos require user-supplied images (no test images are bundled).
The performance benchmark and report scripts use synthetic random inputs.

```bash
# Single-image inference (--image is required)
python models/experimental/atss_swin_l_dyhead/demo/demo_inference.py \
    --image path/to/your/image.jpg

# Batch inference on a directory of images (--image-dir is required)
python models/experimental/atss_swin_l_dyhead/demo/demo_batch.py \
    --image-dir path/to/your/image_directory/

# Performance benchmark (no image needed — uses random tensor input)
python models/experimental/atss_swin_l_dyhead/demo/demo_perf.py

# PCC + performance report (no image needed — uses random tensor input)
python models/experimental/atss_swin_l_dyhead/demo/generate_report.py
```

Output is saved to `atss_swin_l_dyhead/results/` by default. Override with `--output-dir`.

### Usage

```python
from models.experimental.atss_swin_l_dyhead.common import ATSS_CHECKPOINT
from models.experimental.atss_swin_l_dyhead.tt import TtATSSModel

model = TtATSSModel.from_checkpoint(ATSS_CHECKPOINT, device)
results = model.predict(image_tensor, img_shape=(H, W))
```
## Implementation Status

- [x] Phase 0: Reference implementations (backbone, FPN, DyHead, head, postprocess)
- [x] Phase 1: Swin-L backbone (TTNN) -- from standalone swin_l module
- [x] Phase 2: FPN neck (TTNN)
- [x] Phase 3: DyHead -- hybrid (scale/task attention on TTNN, DCNv2 spatial on CPU)
- [x] Phase 4: ATSS Head (TTNN)
- [x] Phase 5: Post-processing (CPU -- anchors, bbox decode, NMS)
- [x] Phase 6: Full model integration (hybrid TTNN + PyTorch)

### Next TODOs
- Full DyHead on TTNN (requires native TTNN DCNv2 kernel)
- Optimization (sharding, precision tuning, L1 memory persistence)
