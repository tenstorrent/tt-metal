# RF-DETR Medium — Detection (TTNN)

TTNN implementation of **RF-DETR Medium** ([ICLR 2026](https://arxiv.org/abs/2511.09554)) real-time
object detection model from [Roboflow](https://github.com/roboflow/rf-detr).
Input 576×576; COCO 80 classes; 33,687,458 params.

| Metric | Value |
|---|---|
| COCO AP50:95 | 54.7 |
| COCO AP50 | 73.6 |
| Latency (T4 FP16 bs1) | 4.4 ms (**~227 FPS**) |
| Parameters | 33,687,458 |
| Resolution | 576×576 |

## Architecture

```
Input Image [1, 3, 576, 576]
  │
  ▼
DINOv2-ViT-S Backbone (TTNN)          22,097,664 params
  │  encoder: "dinov2_windowed_small"
  │  hidden=384, heads=6, layers=12, patch_size=16
  │  NO register tokens (num_register_tokens=0)
  │  CLS token only → 1 CLS + 324 patches = 325 tokens/window
  │  2×2 = 4 windows → windowed shape: [B*4, 325, 384]
  │  Windowed layers: {0,1,2,4,5,7,8,10,11}
  │  Full attention layers: {3,6,9} → reshape to [B, 1300, 384]
  │  Features at stages [3,6,9,12] = after layers [2,5,8,11]
  │  Feature extraction: strip CLS → un-window → [B, 384, 36, 36]
  ▼
MultiScaleProjector (TTNN conv2d)      1,444,864 params
  │  C2f blocks (ttnn.conv2d)             [1, 256, 36, 36]
  │  Fuses 4 backbone features → P4
  │  scale_factors=[1.0]
  ▼
Two-Stage Proposals                    → top-300 query selection
  │  enc_output + cls/bbox + top-K
  ▼
Transformer Decoder (TTNN, 4 layers)   6,084,992 params
  │  Self-attn: 8 heads, SDPA              [1, 300, 256]
  │  Cross-attn: 16 heads, 2 pts (ttnn.grid_sample)
  │  FFN: 256→2048→256
  │  lite_refpoint_refine, bbox_reparam
  ▼
Detection Heads (TTNN)                 → cls (91) + bbox (4)
  │  class_embed: Linear(256, 91)
  │  bbox_embed: MLP(256→256→256→4)
  ▼
Post-processing (Host)                 → sigmoid, top-K, box decode
```

### DINOv2 Backbone Windowing Detail

```
Embedding:
  [B, 3, 576, 576] → patch_embed → [B, 1296, 384]
  Prepend CLS → [B, 1297, 384], add pos encoding
  Window partition → [B*4, 325, 384]  (1 CLS + 324 patches per window)

Layers (tensor always [B*4, 325, 384] between layers):
  Layer 0:  windowed attention  [B*4, 325, 384]
  Layer 1:  windowed attention  [B*4, 325, 384]
  Layer 2:  windowed attention  [B*4, 325, 384]  ← feature extracted (stage3)
  Layer 3:  FULL attention      reshape to [B, 1300, 384], attend, reshape back
  Layer 4:  windowed attention  [B*4, 325, 384]
  Layer 5:  windowed attention  [B*4, 325, 384]  ← feature extracted (stage6)
  Layer 6:  FULL attention      reshape to [B, 1300, 384], attend, reshape back
  Layer 7:  windowed attention  [B*4, 325, 384]
  Layer 8:  windowed attention  [B*4, 325, 384]  ← feature extracted (stage9)
  Layer 9:  FULL attention      reshape to [B, 1300, 384], attend, reshape back
  Layer 10: windowed attention  [B*4, 325, 384]
  Layer 11: windowed attention  [B*4, 325, 384]  ← feature extracted (stage12)
```

## On-Device vs Host

All model computation runs on device. Host is only used for:

| Component | Reason |
|---|---|
| Image preprocessing | NCHW→NHWC pad, one-time |
| Two-stage enc_output + top-K | Float32 precision for stable selection (small) |
| Post-processing | Standard box decode + threshold |

Everything else is pure TTNN on device:

| Component | TTNN Operations |
|---|---|
| DINOv2 backbone | fold+linear, SDPA, layer_norm, reshape/permute for windowing |
| MultiScaleProjector | ttnn.conv2d (C2f blocks), layer_norm, silu |
| Sine positional embedding | ttnn.sin, ttnn.cos, ttnn.div, reshape interleaving |
| ref_point_head MLP | ttnn.linear (2-layer MLP), ttnn.relu |
| Decoder self-attention | linear, matmul, softmax, layer_norm |
| Decoder cross-attention | ttnn.grid_sample (deformable attn), linear, softmax |
| Decoder FFN | linear, relu, layer_norm |
| Detection heads | linear (cls + bbox MLP) |

## Module Structure

```
models/experimental/rfdetr_medium/
├── README.md
├── common.py                           # Constants, model loading
├── reference/
│   ├── __init__.py
│   └── rfdetr_medium.py               # PyTorch staged forward for PCC
├── tt/
│   ├── __init__.py
│   ├── tt_rfdetr.py                   # Full pipeline (TtRFDETR)
│   ├── tt_backbone.py                 # DINOv2-ViT-S with windowed attention
│   ├── tt_projector.py                # MultiScaleProjector (host fallback)
│   ├── tt_decoder.py                  # Transformer decoder (4 layers)
│   ├── tt_detection_heads.py          # cls + bbox heads
│   └── model_preprocessing.py         # Weight loading/conversion
├── tests/
│   ├── pcc/
│   │   ├── conftest.py               # Shared fixtures
│   │   ├── test_ttnn_backbone.py      # Backbone PCC
│   │   ├── test_ttnn_projector.py     # Projector PCC
│   │   ├── test_ttnn_decoder.py       # Decoder PCC
│   │   ├── test_ttnn_heads.py         # Detection heads PCC
│   │   └── test_ttnn_rfdetr_e2e.py    # Full E2E PCC
│   └── perf/
│       └── test_perf.py               # Device performance
└── demo/
    └── demo.py                        # COCO detection demo
```

## Reused Components

| Component | Source | Adaptation |
|---|---|---|
| DINOv2 encoder | `openvla/tt/tt_optimized_openvla_vision.py` | ViT-L→ViT-S (384-dim, 6 heads), added windowed attention |
| Windowed attention | `ttnn/operations/transformer/sdpa_windowed/` + Swin-S | 2×2 window partitioning for DINOv2 blocks |
| Deformable cross-attn | `uniad/tt/ttnn_deformable_attention.py` | Single-level (P4), 16 heads, 2 points |
| Detection heads | Standard `ttnn.linear` | MLP + Linear, straightforward |

## Prerequisites

```bash
cd $TT_METAL_HOME
source python_env/bin/activate
pip install rfdetr
```

## Run PCC Tests

```bash
# Full E2E test
pytest models/experimental/rfdetr_medium/tests/pcc/test_ttnn_rfdetr_e2e.py -v

# Individual component tests
pytest models/experimental/rfdetr_medium/tests/pcc/test_ttnn_backbone.py -v
pytest models/experimental/rfdetr_medium/tests/pcc/test_ttnn_decoder.py -v
pytest models/experimental/rfdetr_medium/tests/pcc/test_ttnn_heads.py -v
pytest models/experimental/rfdetr_medium/tests/pcc/test_ttnn_projector.py -v

# All PCC tests
pytest models/experimental/rfdetr_medium/tests/pcc/ -v
```

## Demo

```bash
# Auto-download COCO images, run TTNN vs PyTorch side-by-side
python models/experimental/rfdetr_medium/demo/demo.py

# Custom image
python models/experimental/rfdetr_medium/demo/demo.py --image path/to/image.jpg

# Lower score threshold
python models/experimental/rfdetr_medium/demo/demo.py --score-thr 0.2

# PyTorch-only (no device required)
python models/experimental/rfdetr_medium/demo/demo.py --skip-ttnn
```

## Usage

```python
import torch
import ttnn
from models.experimental.rfdetr_medium.common import load_torch_model, RFDETR_MEDIUM_L1_SMALL_SIZE
from models.experimental.rfdetr_medium.tt.tt_rfdetr import TtRFDETR
from models.experimental.rfdetr_medium.tt.model_preprocessing import (
    load_backbone_weights, load_decoder_weights, load_detection_head_weights,
)

# Load PyTorch model
torch_model = load_torch_model()

# Open device
device = ttnn.open_device(device_id=0, l1_small_size=RFDETR_MEDIUM_L1_SMALL_SIZE)

# Load weights onto device
backbone_params = load_backbone_weights(torch_model, device)
decoder_params = load_decoder_weights(torch_model, device)
head_params = load_detection_head_weights(torch_model, device)

# Create TTNN model
model = TtRFDETR(
    device=device,
    torch_model=torch_model,
    backbone_params=backbone_params,
    decoder_params=decoder_params,
    head_params=head_params,
)

# Run inference
image = torch.randn(1, 3, 576, 576)  # preprocessed image
result = model.forward(image)

# Get detections
detections = result["detections"]
for det in detections:
    print(f"Boxes: {det['boxes'].shape}, Scores: {det['scores'].shape}")

ttnn.close_device(device)
```

## Implementation Status

- [x] DINOv2-ViT-S backbone — pure TTNN (SDPA, linear, layer_norm, reshape/permute)
- [x] MultiScaleProjector — pure TTNN (ttnn.conv2d, C2f blocks on device)
- [x] Two-stage proposals — enc_output + top-K (float32 host for stable selection)
- [x] Sine positional embedding — pure TTNN (ttnn.sin/cos + reshape interleaving)
- [x] ref_point_head MLP — pure TTNN (2-layer linear + relu)
- [x] Transformer decoder — pure TTNN (self-attn SDPA + deformable cross-attn via ttnn.grid_sample)
- [x] Detection heads — pure TTNN (linear cls + bbox MLP)
- [x] Post-processing (Host)
- [x] PCC tests — component + E2E
- [x] Demo — COCO detection with visualization

### Next TODOs

- Move two-stage to device (requires bfloat16-safe top-K)
- Optimization: sharding, L1 memory persistence, trace, multi-CQ
- Performance benchmarking and comparison
