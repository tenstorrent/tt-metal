# Bark Small — TTNN Implementation

Text-to-audio generation using the [suno/bark-small](https://huggingface.co/suno/bark-small)
model (240M parameters) on Tenstorrent Wormhole hardware via TTNN APIs.

## Architecture

Bark generates speech in 4 stages:

| Stage | Model | Type | Input | Output |
|-------|-------|------|-------|--------|
| 1 | Text-to-Semantic | Causal GPT | BERT tokens | 10k semantic vocab |
| 2 | Semantic-to-Coarse | Causal GPT | Semantic tokens | 2 EnCodec codebooks |
| 3 | Coarse-to-Fine | Non-causal GPT | 2 codebooks | 8 codebooks |
| 4 | EnCodec Decoder | CNN | 8 codebooks | 24kHz mono audio |

Each transformer stage: `hidden_size=768`, `num_heads=12`, `num_layers=12` (~80M params each).

## Quick Start

### Demo
```bash
# Standalone demo
python models/demos/wormhole/bark/demo/demo.py --text "Hello from Tenstorrent!"

# With custom output file
python models/demos/wormhole/bark/demo/demo.py --text "Testing Bark" --output my_audio.wav
```

### Tests
```bash
# Run all tests
pytest models/demos/wormhole/bark/tests/test_bark_model.py -v

# Run specific test class
pytest models/demos/wormhole/bark/tests/test_bark_model.py::TestBarkSemantic -v
pytest models/demos/wormhole/bark/tests/test_bark_model.py::TestBarkPipeline -v

# Run PCC validation
pytest models/demos/wormhole/bark/tests/test_bark_model.py -v -k "pcc"

# Run throughput benchmark
pytest models/demos/wormhole/bark/tests/test_bark_model.py::TestBarkThroughput -v
```

## File Layout

```
models/demos/wormhole/bark/
├── README.md                    # This file
├── tt/
│   ├── bark_gpt.py              # Shared GPT block (attention + MLP + LN)
│   ├── bark_fine.py             # Coarse-to-Fine (non-causal, multi-codebook)
│   └── bark_model.py            # Pipeline orchestrator
├── reference/
│   └── bark_reference.py        # PyTorch reference for PCC comparison
├── demo/
│   └── demo.py                  # Standalone demo script
└── tests/
    └── test_bark_model.py       # Forward pass, PCC, pipeline, throughput tests
```

## Implementation Details

### TTNN Operations Used
- `ttnn.linear` — All projections (QKV, MLP, LM head)
- `ttnn.layer_norm` — Pre-norm in each block + final norm
- `ttnn.gelu` — MLP activation
- `ttnn.add` — Residual connections
- `ttnn.from_torch` / `ttnn.to_torch` — Tensor conversion
- `ttnn.deallocate` — Explicit memory management

### Hybrid Attention
Attention uses a hybrid approach:
- **TTNN**: QKV projection, output projection
- **PyTorch SDPA**: Scaled dot-product attention with causal mask

This ensures correct attention behavior while leveraging TTNN for compute-heavy
linear projections.

### Performance Targets
| Metric | Target | Notes |
|--------|--------|-------|
| Semantic tokens/sec | ≥ 20 | Stage 1 throughput |
| Coarse tokens/sec | ≥ 60 | Stage 2 throughput |
| RTF | < 0.8 | Real-time factor |
| PCC vs PyTorch | ≥ 0.95 | Per-stage accuracy |

## Dependencies

```
transformers>=4.36.0
torch>=2.0
scipy  # For WAV file saving
```

## References

- [Bounty Issue #32069](https://github.com/tenstorrent/tt-metal/issues/32069)
- [Bark Paper](https://arxiv.org/abs/2209.03143)
- [HuggingFace Bark](https://huggingface.co/suno/bark-small)
- [EnCodec](https://github.com/facebookresearch/encodec)
