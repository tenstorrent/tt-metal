# SmolVLA - Vision-Language-Action Model on TT Hardware

SmolVLA is a compact Vision-Language-Action model for robotic manipulation tasks. This implementation runs the full model on Tenstorrent hardware with optimized TTNN operations.

## Model Architecture

```
Image → SigLIP Vision Encoder (12 layers) → Connector → VLM (16 layers) → Expert Layers (16 layers) → Action Head
         ↑                                    ↑           ↑
    TT Optimized                         Spatial      Cross-Attention
                                         Pooling      Conditioning
```

**Key Components:**
- **Vision Encoder**: SigLIP-style ViT with 12 transformer layers
- **Connector**: Spatial pooling + linear projection to VLM dimension
- **VLM Backbone**: 16 transformer layers with K/V caching
- **Expert Layers**: 16 layers with alternating self/cross-attention
- **Action Head**: Flow matching with Euler integration for action prediction

## How to Run

### Prerequisites
Note : Tested only on  Blackhole(p150)
1. Set up TT-Metal environment:
```bash
source python_env/bin/activate
```

2. The model downloads automatically from HuggingFace:
   - Checkpoint: `lerobot/smolvla_base`


### Run Verification Test
```bash
pytest -svv models/experimental/smolvla/tests/test_smol_vla_pcc.py
```

### Run Demo
```bash
python models/experimental/smolvla/demo/demo.py
```

## Performance

| Component | TT Latency | Notes |
|-----------|------------|-------|
| Preprocessing (CPU) | ~77ms | Image tokenization |
| Vision Encoder | ~23ms | 12-layer SigLIP |
| VLM K/V Cache | ~9ms | 16 layers, cached |
| Flow Matching | ~121ms | 10 denoising steps |
| **Total E2E** | **~229ms** | **~4.5 FPS** |

## Verification Results

- **PCC (1-step)**: 0.9988 ✅
- **PCC (5-step)**: 0.9812 ✅
- **PCC (full trajectory)**: 0.9372 ✅
- **Determinism**: 100% (0 variance across runs) ✅
- **Instruction Sensitivity**: Verified ✅

## Files

```
smolvla/
├── README.md                         # This file
├── conftest.py                       # pytest fixtures
├── demo/
│   ├── demo.py                       # Interactive demo
│   └── images/                       # LeRobot sample images
│       ├── lerobot_sample_1.png
│       ├── lerobot_sample_2.png
│       └── lerobot_sample_3.png
├── tests/
│   ├── test_smol_vla_pcc.py          # PCC verification tests
│   └── test_smol_vla_verification.py # Final verification script
└── tt/
    ├── smol_vla.py                   # Main model (~152KB)
    └── ttnn_optimized_vit_smolvla.py # Optimized vision encoder
```

## References

- [SmolVLA Paper](https://huggingface.co/lerobot/smolvla_base)
- [LeRobot Project](https://github.com/huggingface/lerobot)
