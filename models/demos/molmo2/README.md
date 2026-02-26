# Molmo2-8B TTNN Implementation

This directory contains the TTNN implementation of [Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) (Allen AI), a vision-language model with 8.66B parameters.

## Model Overview

Molmo2-8B consists of three sub-systems:

| Sub-system | Parameters | Config |
|---|---|---|
| ViT encoder | 383M (4.4%) | 27 layers (25 used), hidden=1152, heads=16, head_dim=72, patch_size=14 |
| Vision adapter | 88M (1.0%) | Multi-scale concat (layers 18+24), attention pooling, SwiGLU projector |
| Language model | 8,192M (94.6%) | 36 layers, hidden=4096, GQA 32/8, head_dim=128, SwiGLU, QK-norm |

## Supported Devices

- N150 (single chip)
- N300 (2 chips)
- T3K (8 chips)

## Quick Start

```bash
# Set the model path
export HF_MODEL=allenai/Molmo2-8B

# Run the demo
python models/demos/molmo2/demo/demo.py --prompt "Describe this image"
```

## Directory Structure

```
models/demos/molmo2/
├── PLAN.md                       # Implementation plan
├── README.md                     # This file
├── requirements.txt
├── conftest.py
├── reference/
│   └── model.py                  # HuggingFace reference wrapper
├── tt/
│   ├── model_config.py           # Molmo2ModelArgs configuration
│   ├── load_weights.py           # Weight key remapping
│   ├── vision_block.py           # ViT block (LayerNorm + attn + MLP)
│   ├── vision_attention.py       # ViT attention (bidirectional)
│   ├── vision_mlp.py             # GELU MLP
│   ├── vision_transformer.py     # Full ViT encoder
│   ├── image_pooling.py          # Cross-attention pooling
│   ├── image_projector.py        # SwiGLU projector
│   ├── vision_backbone.py        # Combined vision pipeline
│   ├── model.py                  # Text model wrapper
│   └── generator.py              # Prefill/decode loop
├── tests/
│   ├── conftest.py
│   ├── test_vision_block.py
│   ├── test_vision_transformer.py
│   ├── test_image_pooling.py
│   ├── test_image_projector.py
│   ├── test_vision_backbone.py
│   ├── test_language_model.py
│   └── test_full_model.py
└── demo/
    ├── demo.py
    └── sample_prompts/
        └── demo.json
```

## Testing

Run tests with:

```bash
# Run all Molmo2 tests
pytest models/demos/molmo2/tests/ -v

# Run specific test file
pytest models/demos/molmo2/tests/test_vision_block.py -v

# Run with specific device
MESH_DEVICE=N150 pytest models/demos/molmo2/tests/ -v
```

## Implementation Status

- [x] Phase 0: Setup & Reference Model
- [x] Phase 1: Vision Transformer Encoder
- [ ] Phase 2: Vision Adapter
- [ ] Phase 3: Language Model
- [ ] Phase 4: E2E Integration
- [ ] Phase 5: Demo & Performance

## References

- [Molmo2-8B HuggingFace](https://huggingface.co/allenai/Molmo2-8B)
- [Allen AI Molmo Paper](https://arxiv.org/abs/2409.17146)
