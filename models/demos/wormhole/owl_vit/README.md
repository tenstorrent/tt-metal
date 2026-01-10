# OWL-ViT (Open-World Localization Vision Transformer)

## Platforms
- Wormhole (N300)

## Introduction

This demo implements **OWL-ViT** (Vision Transformer for Open-World Localization) using TTNN APIs for zero-shot text-conditioned object detection on Tenstorrent hardware.

OWL-ViT is a powerful model that can detect objects based on natural language queries without category-specific training. It combines:
- **ViT-B/32 Image Encoder**: Processes images into patch embeddings
- **CLIP Text Encoder**: Encodes text queries into embeddings
- **Detection Heads**: Predicts bounding boxes and region-text similarity scores

Reference: [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) (Minderer et al., 2022)

HuggingFace Model: [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32)

## Prerequisites

1. **TT-Metalium / TT-NN Installation**
   - Follow the installation guide at [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

2. **HuggingFace Authentication**
   ```bash
   huggingface-cli login
   # OR
   export HF_TOKEN=<your_token>
   ```

3. **Python Dependencies**
   ```bash
   pip install transformers pillow requests
   ```

## Model Architecture

### Vision Encoder (ViT-B/32)
| Parameter | Value |
|-----------|-------|
| Hidden Size | 768 |
| Attention Heads | 12 |
| Encoder Layers | 12 |
| Intermediate Size | 3072 |
| Image Size | 768×768 |
| Patch Size | 32×32 |
| Num Patches | 576 (24×24) |

### Text Encoder (CLIP)
| Parameter | Value |
|-----------|-------|
| Hidden Size | 512 |
| Attention Heads | 8 |
| Encoder Layers | 12 |
| Vocabulary Size | 49,408 |
| Max Position Embeddings | 16 |

### Detection Heads
| Component | Description |
|-----------|-------------|
| Box Head | 3-layer MLP → 4 outputs (cx, cy, w, h) |
| Class Head | Projects patches to query space, computes similarity |


### 1. Run Tests
```bash
# Run all OWL-ViT tests
pytest models/demos/wormhole/owl_vit/tests/ -v

# Run specific test classes
pytest models/demos/wormhole/owl_vit/tests/test_ttnn_owl_vit.py::TestOwlViTBasicFunctionality -v
pytest models/demos/wormhole/owl_vit/tests/test_ttnn_owl_vit.py::TestOwlViTEndToEnd -v
```

### 3. Run Demo (on TT Hardware)
```bash
# Run the TTNN demo via pytest
pytest models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py -v -s

# Or run directly
python models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py

# Run with custom image and queries
python models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py \
  --image "http://images.cocodataset.org/val2017/000000000285.jpg" \
  --queries "a bear" "video game" "grass" \
  --output "bear_detection.png"
```

### 4. Run PyTorch Reference Demo (CPU)
To compare results with the reference implementation:
```bash
python models/demos/wormhole/owl_vit/demo/demo_owl_vit_pytorch.py
```
Output saved to `demo/outputs/detection_result_pytorch.png`.

### 5. Run Tests (on TT Hardware)
```bash
# Run end-to-end detection test (includes vision encoder PCC validation)
pytest models/demos/wormhole/owl_vit/tests/test_end_to_end.py -v

# Or run directly
python models/demos/wormhole/owl_vit/tests/test_end_to_end.py
```

## Directory Structure

```
models/demos/wormhole/owl_vit/
├── README.md                           # This file
├── __init__.py                         # Package exports
├── reference/
│   └── torch_owl_vit.py               # PyTorch reference implementation
├── tt/
│   └── ttnn_owl_vit.py                # TTNN implementation (modules)
├── tests/
│   ├── test_ttnn_owl_vit.py           # Unit tests
│   └── test_end_to_end.py             # Full detection pipeline + PCC tests
└── demo/
    ├── demo_owl_vit_inference.py      # TTNN inference demo (on device)
    ├── demo_owl_vit_pytorch.py        # PyTorch reference demo (on CPU)
    └── outputs/                        # Demo output images
```

## Implementation Details

### TTNN Optimizations

1. **DRAM Memory Config**: Currently validating with DRAM memory configuration for stability
2. **Fused QKV**: Query, Key, Value projections are fused into a single matrix multiplication
3. **Native TTNN Operations**: Full pipeline implemented using native ttnn ops including embeddings and transformer layers
4. **GELU Fusion**: Feed-forward GELU activation is fused with the linear operation



## Validation Criteria (Stage 1 Bounty)

- [x] Model implementation using TTNN APIs
- [x] Runs on N300 hardware
- [x] Vision encoder produces valid output
- [x] Full end-to-end detection with text encoder and heads
- [x] Output verifiable (region-text similarity scores, visualization)

## Current Status

**Vision Encoder**: Running on TT hardware
- All 12 transformer encoder layers execute on device
- Output shape: [1, 577, 768] matches PyTorch

**Text Encoder**: Running on TT hardware
- All 12 transformer encoder layers execute on device
- Embeddings computed via ttnn.embedding
- Output shape: [batch, 16, 512]

**Detection Heads**: Running on TT hardware
- Box head produces 576 box predictions [batch, 576, 4]
- Class head computes region-text similarity [batch, 576, num_queries]

**End-to-End Pipeline**:
- All components run on TT device
- Objects detected with high confidence scores
- Full detection pipeline functional

## Known Limitations

1. **Batch Size**: Currently optimized for batch size 1
2. **Image Size**: Fixed at 768×768 (OWL-ViT native resolution)

## References

- [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)
- [HuggingFace Model Card](https://huggingface.co/google/owlvit-base-patch32)
- [CLIP Encoder Implementation](../../experimental/tt_dit/encoders/clip/model_clip.py)
