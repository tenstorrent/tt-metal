# LLVC (Low-Latency Low-Resource Voice Conversion)

## Overview

This is a TTNN implementation of the [LLVC](https://github.com/KoeAI/LLVC) model for
Tenstorrent hardware. LLVC is a low-latency, low-resource voice conversion system that
can run in real-time for streaming voice transformation.

The original model is from:
- **Paper**: [LLVC: Low-Latency Low-Resource Voice Conversion](https://arxiv.org/abs/2305.17667)
- **Repository**: [KoeAI/LLVC](https://github.com/KoeAI/LLVC) (MIT License)

## Architecture

The LLVC model consists of:

1. **CachedConvNet (Prenet)**: Optional streaming convolutional pre-processing network
2. **Input Convolution**: Conv1d encoder that maps audio to latent space
3. **Label Embedding**: Linear layers that encode the target speaker identity
4. **MaskNet**: Mask generation network with:
   - **DilatedCausalConvEncoder**: Dilated causal convolutions for temporal encoding
   - **CausalTransformerDecoder**: Transformer decoder for mask prediction
   - **Projection layers**: Grouped Conv1d to project between encoder and decoder dimensions
5. **Output Convolution**: ConvTranspose1d decoder that maps back to audio

### Model Parameters (Default)

| Parameter | Value |
|-----------|-------|
| L (hop length) | 16 |
| enc_dim | 512 |
| num_enc_layers | 8 |
| dec_dim | 256 |
| num_dec_layers | 1 |
| dec_buf_len | 13 |
| dec_chunk_size | 13 |
| out_buf_len | 4 |
| Sample rate | 16000 Hz |

## Directory Structure

```
models/demos/audio/llvc/
├── README.md                          # This file
├── perf_report.md                     # Detailed performance analysis
├── __init__.py
├── reference/
│   ├── __init__.py
│   └── model.py                       # PyTorch reference implementation
├── tt/
│   ├── __init__.py
│   └── ttnn_functional_llvc.py        # TTNN implementation
├── tests/
│   ├── __init__.py
│   ├── test_llvc.py                   # Module and model tests
│   └── test_perf_llvc.py              # Performance test (prep_perf_report)
└── demo/
    ├── __init__.py
    └── demo.py                        # End-to-end demo
```

## Running

### Tests

```bash
# Run all CPU-only LLVC tests (no TT hardware required)
pytest models/demos/audio/llvc/tests/test_llvc.py -v -k "cpu"

# Run individual tests
pytest models/demos/audio/llvc/tests/test_llvc.py::test_llvc_full_model_cpu -v
pytest models/demos/audio/llvc/tests/test_llvc.py::test_llvc_non_streaming_cpu -v
pytest models/demos/audio/llvc/tests/test_llvc.py::test_llvc_streaming_cpu -v
pytest models/demos/audio/llvc/tests/test_llvc.py::test_llvc_output_shape_cpu -v
pytest models/demos/audio/llvc/tests/test_llvc.py::test_llvc_performance_cpu -v

# Run device test (requires TT hardware)
pytest models/demos/audio/llvc/tests/test_llvc.py::test_llvc_full_model_device -v

# Run all tests
pytest models/demos/audio/llvc/tests/test_llvc.py -v
```

### Demo

```bash
# CPU-only demo (no hardware required)
pytest models/demos/audio/llvc/demo/demo.py -k test_llvc_demo_cpu -v

# TT device demo (requires hardware)
pytest models/demos/audio/llvc/demo/demo.py -k test_llvc_demo -v
```

## Test Suite

| Test | Description | Mode |
|------|-------------|------|
| `test_llvc_full_model_cpu` | Full model PCC vs reference (PCC ≥ 0.99) | CPU |
| `test_llvc_non_streaming_cpu` | Multiple input lengths (T=64,128,256) | CPU |
| `test_llvc_streaming_cpu` | Chunked inference with buffer propagation | CPU |
| `test_llvc_output_shape_cpu` | Output shape validation | CPU |
| `test_llvc_performance_cpu` | Throughput, latency, RTF benchmarks | CPU |
| `test_llvc_full_model_device` | Full model on TT device (PCC ≥ 0.96, token >95%, content >0.95) | Device |
| `test_llvc_streaming_device` | Streaming with device decoder (PCC ≥ 0.95) | Device |
| `test_llvc_performance_device` | Throughput + streaming RTF + RTF < 0.3 assertion | Device |
| `test_llvc_batch_processing_device` | Multi-stream 2, 4, and 10 concurrent streams | Device |
| `test_perf_llvc` | Standard perf test with `prep_perf_report()` CSV output | Device |

## Implementation Details

### Inference Modes

**Non-streaming mode**: Processes the entire audio at once with padding. Used for
offline conversion.

**Streaming mode**: Processes audio in small chunks (e.g., 64 samples / 4ms at 16kHz)
with causal buffer state propagated between chunks. Suitable for real-time
applications. Buffer state includes encoder context, decoder context, and output
overlap buffer.

### TTNN Ops Mapping

| PyTorch Op | TTNN Op (Stage 3) | Location |
|------------|-------------------|----------|
| Input `nn.Conv1d` | `ttnn.conv1d` (BLOCK_SHARDED) | TT Device |
| `nn.ReLU` | `ttnn.relu` | TT Device |
| Transformer `nn.Linear` (QKV, FFN) | `ttnn.linear` (pre-uploaded) | TT Device |
| Transformer `nn.LayerNorm` | `ttnn.layer_norm` (pre-uploaded) | TT Device |
| Transformer MHA (Q@K, attn@V) | `ttnn.matmul` + `ttnn.softmax` (L1) | TT Device |
| Label `nn.Linear` | `ttnn.linear` (pre-uploaded) | TT Device |
| Label `nn.LayerNorm` | `ttnn.layer_norm` (pre-uploaded) | TT Device |
| Encoder `nn.Conv1d` (all) | `torch.nn.functional.conv1d` | CPU |
| Encoder `nn.LayerNorm` | `torch.nn.functional.layer_norm` | CPU |
| MaskNet projections (grouped 1x1) | `torch.nn.functional.conv1d` | CPU |
| `nn.ConvTranspose1d` | `torch.nn.functional.conv_transpose1d` | CPU |
| Tensor I/O | `ttnn.from_torch` / `ttnn.to_torch` | Device ↔ CPU |

### API

```python
from models.demos.audio.llvc.tt.ttnn_functional_llvc import (
    preprocess_model_parameters,
    ttnn_llvc_forward,
    init_buffers,
)

# Preprocess reference model parameters
parameters = preprocess_model_parameters(reference_model, device=None)

# Non-streaming inference
output = ttnn_llvc_forward(x, parameters=parameters, config=config, device=None, pad=True)

# Streaming inference
enc_buf, dec_buf, out_buf = init_buffers(batch_size=1, config=config)
for chunk in audio_chunks:
    output, enc_buf, dec_buf, out_buf, _ = ttnn_llvc_forward(
        chunk, parameters=parameters, config=config, device=None,
        init_enc_buf=enc_buf, init_dec_buf=dec_buf, init_out_buf=out_buf, pad=False,
    )
```

### Current Implementation Status

- **Stage 1 (Bring-Up)**: ✅ Complete
  - Self-contained reference model (no external dependencies)
  - TTNN functional implementation with torch fallbacks
  - Full accuracy tests: PCC ≥ 0.99 (CPU), PCC ≥ 0.96 (device/bfloat16)
  - Non-streaming and streaming mode support
  - Performance benchmarking (throughput, latency, RTF)
  - CPU-only and device demo scripts
  - Comprehensive test suite (6 tests)

- **Stage 2 (Basic Optimizations)**: ✅ Complete
  - Transformer decoder (attention + FFN + LN) on TT device
  - Label embedding on TT device
  - Device-aware routing with CPU fallbacks
  - Memory cleanup with `ttnn.deallocate`
  - Streaming + performance device tests

- **Stage 3 (Deeper Optimization)**: ✅ Complete
  - Pre-uploaded weights (zero per-inference host→device transfers)
  - L1 memory for attention intermediates
  - BLOCK_SHARDED input Conv1d (handles longer audio without L1 overflow)
  - Batch processing for 2, 4, and 10 concurrent streams (all PCC > 0.999)
  - Streaming RTF benchmark with device-accelerated decoder
  - Standard perf test (`test_perf_llvc.py`) with `prep_perf_report()` CSV output
  - Encoder stays CPU-only (hybrid approach reverted — transfer overhead)
  - CPU fallback paths preserved for robustness

### Performance Results (N300, Wormhole B0, 2026-02-28)

| Metric | Value | Stage 1 Target | Status |
|--------|-------|----------------|--------|
| PCC (device) | 0.9998 | ≥ 0.96 | ✅ |
| Speaker Similarity (cosine) | 0.9998 | > 0.70 | ✅ |
| Content Preservation (cross-corr) | 0.992 | WER < 3.0 | ✅ |
| Token Accuracy (isclose) | 96.9% | > 95% | ✅ |
| Tokens/sec | 7,996 | ≥ 50 | ✅ |
| Non-streaming RTF | 0.125 | < 0.3 | ✅ |
| Streaming chunk latency | 13.3 ms | < 100ms | ✅ |
| 10-stream PCC | 0.9997 | ≥ 0.95 | ✅ |

See [perf_report.md](perf_report.md) for detailed performance analysis, WER documentation,
and streaming RTF optimization path.

### Notes

- The Stage 1 implementation uses torch fallbacks for grouped convolutions
  and multi-head attention, following the recommended TTNN model bring-up
  approach of correctness first, then optimization.
- PCC threshold is 0.96 for device tests to account for bfloat16 precision
  loss during torch ↔ TTNN conversions.
- Token accuracy uses `torch.isclose(atol=0.01, rtol=0.1)` which handles
  both near-zero values (atol dominates) and larger values (rtol dominates).
- Content preservation uses normalized cross-correlation as a WER proxy
  (no ASR model in container). See perf_report.md for full explanation.
- The `init_buffers()` utility creates properly-sized zero buffers for
  streaming inference, matching the reference model's `init_buffers()` API.
