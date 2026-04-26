# DINO-5scale Swin-L (TTNN)

TTNN implementation of **DINO** (DETR with Improved deNoising anchOr boxes) object detection
with **5-scale** feature pyramid, **Swin-L** backbone, transformer **encoder/decoder**,
and iterative box refinement. Input size 1333×800; COCO 80 classes.

## Architecture

```
Input Image [1, 3, 800, 1333]
  │
  ▼
Swin-L Backbone (TTNN)          → 4 feature maps (stages 0–3)
  │                                [192, 384, 768, 1536] channels
  ▼
ChannelMapper Neck (TTNN)        → 5 feature maps (P2–P6)
  │                                all 256 channels, conv + GroupNorm
  ▼
Pre-Transformer (Host)           → flatten + positional encoding + level embed
  │
  ▼
Encoder (TTNN, 6 layers)        → multi-scale deformable attention + FFN
  │                                ~89K queries across 5 levels
  ▼
Pre-Decoder (Host, float32)      → top-K selection (900 queries) + proposals
  │
  ▼
Decoder (TTNN, 6 layers)        → self-attention + cross-attention + FFN
  │                                900 queries, iterative box refinement
  ▼
Detection Heads (TTNN)           → per-layer: cls (80), bbox (4)
  │
  ▼
Post-processing (Host)           → sigmoid, NMS → bboxes, scores, labels
```

## Host Fallbacks

The following operations run on the host CPU (not on device):

| Component | Operation | Reason |
|---|---|---|
| Pre-Transformer | Sine positional encoding, level embed, flatten | Lightweight, one-time per inference |
| Encoder (MSDeformAttn) | Sampling location computation, attention weight softmax | 6D reshape not supported in TTNN |
| Encoder (MSDeformAttn) | `grid_sample` bilinear interpolation | Uses `ttnn.grid_sample` but values transferred via host |
| Pre-Decoder | `memory_trans_fc`, `layer_norm`, `cls_branches[6]`, `reg_branches[6]`, `top-K` | Float32 precision required — top-K is extremely sensitive to bfloat16 rounding |
| Decoder | Reference point refinement (`inverse_sigmoid` + delta) | Small tensor, requires float32 |
| Decoder | Sine query embed generation | Small tensor, computed per-layer |
| Detection Heads | Collect cls/bbox outputs across layers | `torch.stack` on host |
| Post-processing | Sigmoid, NMS, bbox decode | Standard CPU post-processing |

## Module Structure

```
models/experimental/dino_5scale_swin_l/
├── common.py                           # Constants (input size, num_queries, etc.)
├── reference/                          # PyTorch reference + config
│   ├── dino_staged_forward.py         #   Staged forward for PCC comparison
│   ├── swin_l_reference.py            #   Swin-L reference wrapper
│   ├── dino_5scale_swin_l.py          #   mmdet config file
│   └── infer.py                       #   mmdet reference inference script
├── tt/                                 # TTNN implementations
│   ├── tt_dino.py                     #   Full pipeline (TtDINO)
│   ├── tt_neck.py                     #   ChannelMapper (conv + GroupNorm)
│   ├── tt_encoder.py                  #   Transformer encoder + MSDeformAttn
│   ├── tt_decoder.py                  #   Transformer decoder + heads
│   └── model_preprocessing.py         #   Weight loading (neck, encoder, decoder)
├── tests/pcc/                          # PCC validation tests
│   ├── conftest.py                    #   Shared fixtures
│   ├── test_ttnn_dino_e2e.py          #   Full E2E (real image, stage-by-stage PCC)
│   ├── test_ttnn_backbone.py          #   Swin-L backbone PCC
│   ├── test_ttnn_neck.py              #   ChannelMapper neck PCC
│   ├── test_ttnn_encoder.py           #   Encoder PCC
│   ├── test_ttnn_decoder.py           #   Decoder PCC
│   ├── test_ttnn_heads.py             #   Detection heads PCC
│   ├── test_ttnn_swin_attention.py    #   Swin attention PCC
│   ├── test_ttnn_swin_block.py        #   Swin block PCC
│   ├── test_ttnn_swin_mlp.py          #   Swin MLP PCC
│   └── test_ttnn_swin_patch_merge.py  #   Swin patch merge PCC
├── tests/reference/
│   └── test_reference_staged_forward.py  # PyTorch reference validation
├── demo/
│   └── demo.py                        #   TTNN vs PyTorch side-by-side comparison
└── checkpoints/dino_5scale_swin_l/
    └── *.pth                          #   Model weights (downloaded separately)

models/experimental/swin_l/             # Reusable Swin-L backbone (separate module)
├── tt/
│   ├── tt_backbone.py                 #   TtSwinLBackbone
│   ├── tt_swin_attention.py           #   Shifted window attention
│   ├── tt_swin_mlp.py                 #   MLP
│   ├── tt_swin_block.py               #   Transformer block
│   ├── tt_swin_patch_merge.py         #   Patch merging (downsample)
│   └── model_preprocessing.py         #   Backbone weight loading + attn masks
└── common.py
```

## Quick Start

### Prerequisites

```bash
cd $TT_METAL_HOME
source python_env/bin/activate
export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages
```

### Checkpoint

```bash
pip install openmim && mim install mmdet
mkdir -p models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l
mim download mmdet --config dino-5scale_swin-l_8xb2-36e_coco \
    --dest models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l
```

### Run PCC Tests

```bash
# Full E2E test (real image, all stages)
pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_dino_e2e.py -v

# Individual component tests
pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_backbone.py -v
pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_neck.py -v
pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_encoder.py -v
pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_decoder.py -v

# All tests
pytest models/experimental/dino_5scale_swin_l/tests/pcc/ -v
```

### Demo

```bash
# TTNN vs PyTorch side-by-side on 3 COCO images (auto-downloaded)
python models/experimental/dino_5scale_swin_l/demo/demo.py

# Custom image
python models/experimental/dino_5scale_swin_l/demo/demo.py --image path/to/image.jpg

# Lower score threshold
python models/experimental/dino_5scale_swin_l/demo/demo.py --score-thr 0.2
```

Output: side-by-side comparison images saved to `demo/` folder.

### Usage

```python
import ttnn
from models.experimental.dino_5scale_swin_l.tt import (
    TtDINO, TtSwinLBackbone,
    load_backbone_weights, load_neck_weights,
    load_encoder_weights, load_decoder_weights, compute_attn_masks,
)

device = ttnn.open_device(device_id=0, l1_small_size=32768)
backbone_params = load_backbone_weights(ckpt_path, device)
neck_params = load_neck_weights(ckpt_path, device)
encoder_params = load_encoder_weights(ckpt_path, device)
decoder_params = load_decoder_weights(ckpt_path, device)
attn_masks = compute_attn_masks(800, 1333, 4, 12, device)

model = TtDINO(
    encoder_params=encoder_params, decoder_params=decoder_params,
    device=device, backbone_params=backbone_params,
    neck_params=neck_params, attn_masks=attn_masks,
    num_queries=900, num_classes=80, num_levels=5,
    embed_dims=256, num_heads=8, num_points=4,
    encoder_num_layers=6, decoder_num_layers=6,
    pe_temperature=20, embed_dim=192,
    depths=(2, 2, 18, 2), backbone_num_heads=(6, 12, 24, 48),
    window_size=12, in_channels=(192, 384, 768, 1536),
)

result = model.forward_image(image_tensor)  # [1, 3, 800, 1333] float32
detections = TtDINO.postprocess(
    result["all_cls_scores"][-1], result["all_bbox_preds"][-1],
    img_shape=(800, 1333), score_thr=0.3,
)
```

## PCC Results (cats_remotes.jpg)

| Component | PCC |
|---|---|
| Backbone stage 0 (C2) | 0.9999 |
| Backbone stage 1 (C3) | 0.9998 |
| Backbone stage 2 (C4) | 0.9978 |
| Backbone stage 3 (C5) | 0.9994 |
| Neck level 0 (P2) | 0.9979 |
| Neck level 1 (P3) | 0.9977 |
| Neck level 2 (P4) | 0.9704 |
| Neck level 3 (P5) | 0.9716 |
| Neck level 4 (P6) | 0.9879 |
| Encoder memory | 0.9990 |
| Top-K overlap | 821/900 (91.2%) |
| Decoder layer 0 (matched-query) | 0.9850 |
| Decoder layer 5 (matched-query) | 0.9702 |
| Detection match rate (score>0.3) | 5/5 (100%) |
| Avg detection IoU | 0.996 |
| Avg score difference | 0.0076 |

## Implementation Status

- [x] Swin-L backbone (TTNN) — from standalone `swin_l` module
- [x] ChannelMapper neck (TTNN) — conv2d + GroupNorm
- [x] Transformer encoder (TTNN) — MSDeformAttn with batched grid_sample
- [x] Transformer decoder (TTNN) — self-attn + cross-attn + FFN + box refinement
- [x] Detection heads (TTNN) — cls + reg branches
- [x] Pre-decoder (Host float32) — top-K selection
- [x] Post-processing (Host) — sigmoid, NMS
- [x] Full E2E pipeline + demo

### Next TODOs

- Move MSDeformAttn sampling location computation fully on-device
- Move pre-transformer (positional encoding, flatten) on-device
- Optimization (sharding, L1 memory persistence, multi-CQ)
