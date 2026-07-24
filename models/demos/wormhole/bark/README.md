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
# Standalone demo (greedy decoding)
python models/demos/wormhole/bark/demo/demo.py --text "Hello from Tenstorrent!"

# With top-k sampling for more natural speech variety
python models/demos/wormhole/bark/demo/demo.py --text "Hello from Tenstorrent!" --top-k 50 --temperature 0.8

# Demo writes bark_output.wav in the current working directory
python models/demos/wormhole/bark/demo/demo.py --text "Testing Bark"
```

### Tests
```bash
# CI smoke test (standard entry point)
pytest models/demos/wormhole/bark/tests/test_bark_demo.py -v

# CI performance test (generates perf report CSV)
pytest models/demos/wormhole/bark/tests/test_bark_perf.py -v

# Run all model tests
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
├── PERFORMANCE_REPORT.md        # Throughput, RTF, PCC results
├── MEMORY_BUDGET.md             # L1/DRAM memory budget per operation
├── tt/
│   ├── bark_gpt.py              # Core GPT block (attention + MLP + LN)
│   ├── bark_fine.py             # Coarse-to-Fine (non-causal, multi-codebook)
│   ├── bark_model.py            # Pipeline orchestrator & generation loops
│   ├── bark_constants.py        # Shared vocab/token constants (single source of truth)
│   ├── bark_long_text.py        # Long text support (500+ chars, chunking)
│   ├── bark_voice_presets.py    # Voice preset loading & caching
│   ├── bark_batch.py            # Batch audio generation
│   └── bark_pipeline_overlap.py # Streaming pipeline overlap analysis
├── reference/
│   └── bark_reference.py        # PyTorch reference for PCC comparison
├── demo/
│   └── demo.py                  # Standalone demo script
└── tests/
    ├── test_bark_demo.py        # CI entry point — standard test_demo()
    ├── test_bark_perf.py        # CI performance test with prep_perf_report
    ├── test_bark_model.py       # Forward pass, PCC, pipeline, throughput tests
    ├── test_bark_reference_parity.py  # CPU-only token pipeline validation
    ├── run_bark_e2e.py          # End-to-end text→audio test suite
    ├── validate_token_accuracy.py # PCC + top-1 validation per stage
    ├── profile_bark.py          # Per-stage throughput profiler
    └── debug_pcc.py             # Per-layer PCC divergence debugger
```

### Optimization Details (Stages 2 & 3)

The implementation uses TTNN ops for all transformer computation:
- **Full TTNN Attention**: All attention masking and scaling occur on-device via `ttnn.transformer.scaled_dot_product_attention` (prefill) or explicit matmul (decode). No `to_torch` calls inside transformer blocks.
- **Pre-Allocated KV Cache**: Uses `ttnn.kv_cache.fill_cache_for_user_` (prefill) and `ttnn.kv_cache.update_cache_for_token_` (decode) for O(n) write-in-place updates instead of O(n²) concat. Cache stored in DRAM.
- **On-Device Logits Masking**: Pre-created suppression masks applied via `ttnn.add` + `ttnn.argmax` on device. Host sync only for EOS detection (every 4 steps semantic, every step coarse for codebook-pair alignment).
- **Stage 3 Persistent Tokens**: The fine acoustics stage maintains all 8 codebooks on-device, with on-device `ttnn.argmax` for codebook prediction.
- **Device-Side Decode Embeddings**: Embedding weights are pre-transferred to device DRAM at model load. During decode (seq_len=1), `ttnn.embedding` performs the lookup on-device, eliminating per-token CPU→device transfers. Falls back to CPU `nn.Embedding` if NCRISC compilation fails.
- **Top-k Sampling**: Optional temperature-scaled top-k sampling for more natural speech variety. Temperature scaling is applied on-device before host-side multinomial.
- **Compute Grid Tuning**: Configured to utilize the available compute grid on Wormhole (8×7 on N300, 8×8 on N150).
- **Chunked Coarse→Fine Processing**: Fine model processes coarse tokens in configurable chunks (default 100 frames), reducing peak memory. True concurrent overlap requires two devices; on single N300 the stages run sequentially. Full multi-device overlap is a follow-up.
- **Intermediate Tensor Cleanup**: Transposed key tensors, pre-norm hidden states, and KV cache are explicitly deallocated to minimize L1/DRAM pressure.
- **Operator Fusion**: MLP projections via `ttnn.linear`, GELU_NEW activation decomposed on-device (`x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715x³)))`).

### TTNN Operations Used
- `ttnn.linear` — Projections and fused MLP transformations
- `ttnn.transformer.scaled_dot_product_attention` — On-device attention
- `ttnn.layer_norm` — Pre-norm in each block + final norm
- `ttnn.embedding` — On-device lookups (decode path; semantic/coarse use CPU `nn.Embedding` for prefill, device-side `ttnn.embedding` for decode. Falls back to CPU if NCRISC compiler bug is triggered, see [#32069](https://github.com/tenstorrent/tt-metal/issues/32069))
- `ttnn.add` / `ttnn.slice` / `ttnn.reshape` — Tensor manipulation
- `ttnn.argmax` / `ttnn.multiply` / `ttnn.softmax` — Token selection & sampling
- `ttnn.deallocate` — Explicit memory management

### Performance Targets
| Metric | Target | Status |
|--------|--------|--------|
| Semantic tokens/sec | >= 20 | PASS - 92.0 tok/s (4.6x target) |
| Coarse tokens/sec | >= 60 | PASS - 67.0 tok/s |
| Fine tokens/sec | >= 60 | PROJECTED - ~600 tok/s (pending CI) |
| RTF | < 0.8 | PROJECTED - ~0.70 (pending N300 CI validation) |
| PCC vs PyTorch | >= 0.95 | PASS - All stages > 0.999 |

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
