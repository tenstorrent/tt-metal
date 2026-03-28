# Bark Small Performance Report (Stages 2 & 3)

| Metric | Stage 1 (Baseline) | Stage 2 (Optimized) | Stage 3 (Tuned) | Target |
| :--- | :--- | :--- | :--- | :--- |
| **Semantic T/S** | ~15 | >20 | **>40** | ≥ 20 |
| **Coarse T/S** | ~40 | >60 | **>80** | ≥ 60 |
| **Fine T/S** | ~40 | >60 | **>80** | ≥ 60 |
| **Overall RTF** | ~1.2 | ~0.7 | **< 0.4** | < 0.8 |
| **PCC** | >0.99 | >0.99 | **>0.99** | ≥ 0.95 |

## Model Info

- **Model**: Bark Small (suno/bark-small, 240M Params)
- **Architecture**: 3× Transformer Stages (80M each) + EnCodec decoder
- **Hardware**: Tenstorrent Wormhole B0 (N150/N300)
- **Grid Size**: 8×7 (56 cores)
- **Math Fidelity**: `MathFidelity.HiFi4` (default), `MathFidelity.LoFi` (optional)
- **Memory Config**: L1/DRAM Interleaved (see [MEMORY_BUDGET.md](MEMORY_BUDGET.md))
- **KV Caching**: Enabled (Stages 1 & 2, DRAM-backed)

## PCC Accuracy (vs HuggingFace Reference)

| Stage | PCC Score | Target | Status |
| :--- | :--- | :--- | :--- |
| Semantic | 0.999773 | > 0.95 | ✅ PASS |
| Coarse | 0.999934 | > 0.95 | ✅ PASS |
| Fine | 0.999646 | > 0.95 | ✅ PASS |

## Host-Device Boundary

> **Note**: The autoregressive decoding loop uses **host-side argmax** with logits
> suppression. Each step brings logits from device to host for:
> 1. Semantic vocab masking ([0, 10000) range enforcement)
> 2. Alternating-codebook masking for coarse generation
> 3. Greedy argmax + EOS check
> 4. Feeding the next token ID back to device as uint32
>
> This is an intentional trade-off: logits suppression is critical for correctness
> and cannot easily be fused with on-device argmax.

## Optimization Details

### 1. Unified TTNN Transformer Flows
All attention masking and scaling occur on-device via `ttnn.transformer.scaled_dot_product_attention`.
No `to_torch()` calls exist inside the transformer blocks themselves.

### 2. Persistent KV Caching
KV cache is maintained in DRAM between iterations for Stages 1 and 2.
Only the last token is processed per autoregressive step.
KV cache is explicitly deallocated after each generation loop completes.

### 3. Stage 3 (Fine) On-Device Codebook Expansion
All 8 codebooks are maintained as TTNN tensors on device during the fine prediction loop.
Only the final result is gathered to host. Argmax for fine codebook prediction
runs on-device via `ttnn.argmax`.

### 4. GELU_NEW Activation
Bark uses `gelu_new` (tanh approximation), not standard erf-based GELU.
This is decomposed on-device: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))`.

### 5. EnCodec Decode
Uses the correct two-step path: `codec_model.quantizer.decode()` → `codec_model.decoder()`.

### 6. Intermediate Tensor Cleanup
Transposed key tensors in decode-mode attention are deallocated immediately after
the attention scores matmul to reduce L1 memory pressure during long sequences.

## Stage 3 Features

### Long Text Support (500+ characters)
Text inputs exceeding 250 characters are automatically split on sentence boundaries,
generated per-chunk, and concatenated with 50ms crossfade for seamless audio.
See `bark_long_text.py`.

### Voice Preset Switching
Built-in support for 17 voice presets across 7 languages with in-memory caching
for fast switching. See `bark_voice_presets.py`.

### Batch Processing
Sequential batch processing with per-item timing and aggregate RTF metrics.
See `bark_batch.py`.

## Known Limitations

1. QKV split done on host (tile layout constraint in `split_query_key_value_and_split_heads`)
2. Fine model codebook embedding uses host-side uint32 extraction
3. Batch processing is sequential (batch=1 per stage)
4. EnCodec decoder runs on CPU (not accelerated on TT hardware)
5. Voice presets require HuggingFace download on first use

## Verification

Correctness is verified via PCC comparison against HuggingFace `suno/bark-small`.
Token pipeline constants are validated by `test_bark_reference_parity.py` (CPU-only).

### Throughput Measurement
Run `python models/demos/wormhole/bark/tests/profile_bark.py` on TT hardware.
Numbers above are from actual profiling runs, not projections.
