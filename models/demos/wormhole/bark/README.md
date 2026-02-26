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
| 4 | EnCodec Decoder | CNN/Device | 8 codebooks | 24kHz mono audio |

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
│   ├── bark_gpt.py              # Core GPT block (attention + MLP + LN)
│   ├── bark_fine.py             # Coarse-to-Fine (non-causal, multi-codebook)
│   └── bark_model.py            # Pipeline orchestrator & On-device loops
├── reference/
│   └── bark_reference.py        # PyTorch reference for PCC comparison
├── demo/
│   └── demo.py                  # Standalone demo script
└── tests/
    └── test_bark_model.py       # Forward pass, PCC, pipeline, throughput tests
```

### Optimization Details (Stages 2 & 3)

The implementation has been fully optimized for Tenstorrent hardware:
- **Full TTNN Attention**: Eliminated all `to_torch` calls in the transformer blocks. All attention masking and scaling occur on-device.
- **On-Device KV Caching**: Integrated persistent KV caches for stages 1 and 2, drastically reducing the compute requirements for autoregressive generation.
- **On-Device Loops**: Generation loops for stages 1 and 2 run mostly on-device; `ttnn.argmax` is used on-device, with only a scalar EOS-check token transferred to host per iteration.
- **Stage 3 Persistent Tokens**: The fine acoustics stage maintains all 8 codebooks on-device as a list of tensors, eliminating host-side synchronization during the codebook expansion process.
- **Compute Grid Tuning**: Configured to utilize the available compute grid on Wormhole (e.g., 8x8 on N150).
- **LoFi Math Fidelity**: Optimized math fidelity settings for increased throughput with negligible accuracy loss.
- **Operator Fusion**: Fused MLP projections and GELU activations using `ttnn.linear(activation="gelu")`.

### TTNN Operations Used
- `ttnn.linear` — Projections and Fused MLP transformations
- `ttnn.transformer.scaled_dot_product_attention` — On-device attention
- `ttnn.layer_norm` — Pre-norm in each block + final norm
- `ttnn.embedding` — On-device lookups for tokens and positions
- `ttnn.add` / `ttnn.slice` / `ttnn.reshape` — Tensor manipulation
- `ttnn.deallocate` — Explicit memory management

### Performance Targets
| Metric | Target | Status |
|--------|--------|--------|
| Semantic tokens/sec | ≥ 20 | ✅ Optimized |
| Coarse tokens/sec | ≥ 60 | ✅ Optimized |
| RTF | < 0.8 | ✅ Target Met |
| PCC vs PyTorch | ≥ 0.95 | ✅ Verified |

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
