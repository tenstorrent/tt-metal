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

### 1. Run Reference PyTorch Demo
Validate the model works before running on TT hardware:

```bash
python models/demos/wormhole/owl_vit/reference/torch_owl_vit.py
```

### 2. Run Tests
```bash
# Run all OWL-ViT tests
pytest models/demos/wormhole/owl_vit/tests/ -v

# Run specific test classes
pytest models/demos/wormhole/owl_vit/tests/test_ttnn_owl_vit.py::TestOwlViTBasicFunctionality -v
pytest models/demos/wormhole/owl_vit/tests/test_ttnn_owl_vit.py::TestOwlViTEndToEnd -v
```

### 3. Run Demo
```bash
pytest models/demos/wormhole/owl_vit/demo/demo_owl_vit_inference.py -v -s
```

### 4. Run on TT Hardware
```bash
pytest models/demos/wormhole/owl_vit/tests/test_device_inference.py -v
```


## Directory Structure

```
models/demos/wormhole/owl_vit/
├── README.md                           # This file
├── __init__.py                         # Package exports
├── reference/
│   └── torch_owl_vit.py               # PyTorch reference implementation
├── tt/
│   ├── ttnn_owl_vit.py                # TTNN implementation (modules)
│   └── ttnn_optimized_sharded_vit_wh.py # Base ViT implementation
├── tests/
│   ├── test_ttnn_owl_vit.py           # Unit tests
│   ├── test_device_inference.py       # Vision encoder device test
│   └── test_end_to_end.py             # Full detection pipeline test
└── demo/
    ├── demo_owl_vit_inference.py      # Inference demo
    └── outputs/                        # Demo output images
```

## Implementation Details

### TTNN Optimizations

1. **Block Sharding**: Vision encoder uses 8×8 core grid for efficient parallel computation
2. **Fused QKV**: Query, Key, Value projections are fused into a single matrix multiplication
3. **Memory Management**: Uses L1 sharded memory for inter-operation data transfer
4. **GELU Fusion**: Feed-forward GELU activation is fused with the linear operation


## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Batch Size | 1 | 1 ✅ |
| Vision Encoder PCC | > 0.8 | 0.86 ✅ |
| Images/sec | TBD | ~1s (vision + text + heads) |

## Validation Criteria (Stage 1 Bounty)

- [x] Model implementation using TTNN APIs
- [x] Runs on N300 hardware
- [x] Vision encoder produces valid output (PCC: 0.86 vs PyTorch)
- [x] Full end-to-end detection with text encoder and heads
- [x] Output verifiable (region-text similarity scores, visualization)

## Current Status

**Vision Encoder**: ✅ Running on TT hardware
- All 12 transformer encoder layers execute on device
- Output shape: [1, 577, 768] matches PyTorch
- Pearson Correlation Coefficient: 0.86

**Text Encoder**: ✅ Running on TT hardware
- All 12 transformer encoder layers execute on device
- Embeddings computed via ttnn.embedding
- Output shape: [batch, 16, 512]

**Detection Heads**: ✅ Running on TT hardware
- Box head produces 576 box predictions [batch, 576, 4]
- Class head computes region-text similarity [batch, 576, num_queries]

**End-to-End Pipeline**: ✅ Complete
- All components run on TT device
- Objects detected with high confidence scores
- Full detection pipeline functional

## Known Limitations

1. **Batch Size**: Currently optimized for batch size 1
2. **Image Size**: Fixed at 768×768 (OWL-ViT native resolution)
3. **Numerical Precision**: Using bfloat16/bfloat8 causes some deviation from fp32 PyTorch
4. **Box Calibration**: Detection heads need full logit_scale/logit_shift for exact PyTorch match

## References

- [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)
- [HuggingFace Model Card](https://huggingface.co/google/owlvit-base-patch32)
- [ViT TTNN Tech Report](../../tech_reports/ViT-TTNN/vit.md)
- [CLIP Encoder Implementation](../../experimental/tt_dit/encoders/clip/model_clip.py)
