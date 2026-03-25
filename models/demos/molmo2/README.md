# Molmo2-8B TTNN Implementation

## Introduction

This directory contains the TTNN implementation of [Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) (Allen AI), a vision-language model with 8.66B parameters. The implementation supports efficient inference on Tenstorrent hardware with full tracing support for optimized performance.

## Performance (T3K - 8 devices)

| Metric | Performance |
|--------|-------------|
| Vision processing (traced) | ~86ms |
| Prefill TTFT | ~85ms |
| Decode throughput (traced) | **35.6 tok/s** (~28ms/token) |
| Decode throughput (no trace) | 5.5 tok/s (~181ms/token) |
| Tracing speedup | 6.5x decode, 25x vision |

## Model Architecture

Molmo2-8B consists of three sub-systems:

| Sub-system | Parameters | Configuration |
|------------|------------|---------------|
| ViT Encoder | 383M (4.4%) | 27 layers (25 used), hidden=1152, heads=16, head_dim=72, patch_size=14 |
| Vision Adapter | 88M (1.0%) | Multi-scale concat (layers 18+24), cross-attention pooling, SwiGLU projector |
| Language Model | 8,192M (94.6%) | 36 layers, hidden=4096, GQA 32/8, head_dim=128, SwiGLU MLP, QK-norm |

## Supported Devices

| Device | Configuration |
|--------|---------------|
| N150 | Single chip |
| N300 | 2 chips |
| T3K | 8 chips (recommended) |

## Prerequisites

1. Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
2. Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
3. Install additional dependencies:

```bash
pip install -r models/demos/molmo2/requirements.txt
```

## How to Run

### Basic Usage

```bash
# Run with default image (dog on skateboard)
python models/demos/molmo2/demo/demo.py --prompt "<|image|> Describe this image."

# Run with custom image
python models/demos/molmo2/demo/demo.py --prompt "<|image|> What do you see?" --image path/to/image.jpg
```

### Optimized Execution (Recommended)

For best performance, enable tracing:

```bash
# Full tracing - 35.6 tok/s decode, 86ms vision
python models/demos/molmo2/demo/demo.py \
    --prompt "<|image|> Describe this image in detail." \
    --use-trace \
    --use-decode-trace \
    --use-vision-trace \
    --max-tokens 100
```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--prompt` | Input prompt (use `<\|image\|>` placeholder for image) |
| `--image` | Path to input image (default: sample dog image) |
| `--max-tokens` | Maximum tokens to generate (default: 100) |
| `--use-trace` | Enable prefill tracing |
| `--use-decode-trace` | Enable decode tracing |
| `--use-vision-trace` | Enable vision backbone tracing |
| `--num-layers` | Number of decoder layers (default: all 36) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_MODEL` | HuggingFace model path (default: `allenai/Molmo2-8B`) |
| `MESH_DEVICE` | Device configuration: `N150`, `N300`, or `T3K` |

## Directory Structure

```
models/demos/molmo2/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── conftest.py                   # Pytest configuration
├── reference/
│   └── model.py                  # HuggingFace reference wrapper
├── tt/
│   ├── model_config.py           # Molmo2ModelArgs configuration
│   ├── load_weights.py           # Weight loading utilities
│   │
│   │   # Vision components
│   ├── vision_attention.py       # ViT multi-head attention
│   ├── vision_mlp.py             # ViT GELU MLP
│   ├── vision_layernorm.py       # ViT LayerNorm
│   ├── vision_block.py           # ViT transformer block
│   ├── vision_transformer.py     # Full ViT encoder (25 layers)
│   ├── image_pooling.py          # Cross-attention pooling
│   ├── image_projector.py        # SwiGLU projector to LM dim
│   ├── vision_backbone.py        # Combined vision pipeline
│   │
│   │   # Text/Language components
│   ├── text_attention.py         # GQA with QK-norm, fused QKV
│   ├── text_mlp.py               # SwiGLU MLP
│   ├── text_rmsnorm.py           # RMSNorm
│   ├── text_rotary_emb.py        # RoPE embeddings
│   ├── text_rotary_setup.py      # RoPE setup utilities
│   ├── text_block.py             # Decoder block
│   ├── text_model.py             # Full text decoder
│   │
│   └── molmo2_model.py           # Combined VLM model
├── tests/
│   └── *.py                      # Unit and integration tests
└── demo/
    └── demo.py                   # Interactive demo
```

## Implementation Details

### Key Optimizations

1. **Fused QKV Matmul**: Single matmul for Q/K/V projections with `nlp_create_qkv_heads_decode`
2. **TTNN-Native RoPE**: Uses `ttnn.experimental.rotary_embedding_llama` for efficient rotary embeddings
3. **Vision Tracing**: `forward_ttnn()` method enables full vision backbone tracing with `ttnn.embedding` for gather ops
4. **Pre-concatenated Embeddings**: Token embeddings pre-concatenated at init for trace compatibility
5. **Tensor Parallelism**: Automatic sharding across devices for multi-chip configurations

### Tracing Architecture

The implementation supports three independent traces:
- **Vision trace**: ViT encoder + pooling + projection (~86ms)
- **Prefill trace**: Text model prefill with KV cache fill (~85ms TTFT)
- **Decode trace**: Autoregressive token generation (~28ms/token)

## Testing

```bash
# Run all Molmo2 tests
pytest models/demos/molmo2/tests/ -v

# Run specific test
pytest models/demos/molmo2/tests/test_vision_backbone.py -v

# Run with specific device
MESH_DEVICE=T3K pytest models/demos/molmo2/tests/ -v
```

## Known Limitations

1. **Decode RoPE**: Uses PyTorch-based RoPE computation during decode (HEIGHT_SHARDED requirement workaround)
2. **Unified Trace**: Combined vision+prefill trace is disabled due to TTNN trace capture limitations with embedding ops
3. **Weight Precision**: Uses bfloat16 weights during decode to avoid numerical overflow (bfloat8_b causes issues)
4. **Vision trace**: `forward_ttnn` uses the same per-position masked mean and additive SDPA mask as `forward()` / HF (fixed shapes; mask values vary per input).

## Future Optimizations

The following optimizations are identified but not yet implemented:

| Optimization | Benefit | Blocker |
|---|---|---|
| Unified vision+prefill trace | Eliminate trace setup latency | TTNN does not support `ttnn.embedding` writes during trace capture |
| Native decode RoPE via `rotary_embedding_llama` | Reduce decode latency | HEIGHT_SHARDED tensor layout incompatibility |
| Fused QKV in text prefill | Reduce prefill matmuls from 3→1 | None; straightforward to implement |
| bfloat8_b decode weights | Reduce memory bandwidth | Numerical overflow in deep layers; needs investigation |

## References

- [Molmo2-8B on HuggingFace](https://huggingface.co/allenai/Molmo2-8B)
- [Molmo: Open Weights and Open Data (Paper)](https://arxiv.org/abs/2409.17146)
- [Allen AI Molmo Project](https://molmo.allenai.org/)
