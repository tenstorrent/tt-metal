# Bark Small — Performance Report

## Model Overview

| Property | Value |
| :--- | :--- |
| Model | [suno/bark-small](https://huggingface.co/suno/bark-small) |
| Parameters | 240M total (80M × 3 stages) |
| Architecture | GPT-2 style transformer (causal for stages 1-2, non-causal for stage 3) |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Layers | 12 per stage |
| Output | 24 kHz mono audio via EnCodec (8 codebooks) |

## Hardware Configuration

| Property | Value |
| :--- | :--- |
| Device | Tenstorrent Wormhole N300 |
| Compute Grid | 8×7 (56 cores) |
| DRAM | Weights + KV cache |
| L1 | Activations + intermediate tensors |
| Math Fidelity | HiFi4 (fp32 dest accumulation) |

## PCC Validation Results

All scores measured against HuggingFace `suno/bark-small` PyTorch reference.

| Stage | Forward PCC | Top-1 Agreement | Per-Layer Min PCC |
| :--- | :--- | :--- | :--- |
| Semantic | 0.999773 | >99% | 0.999864 (L11) |
| Coarse | 0.999934 | >99% | >0.999 |
| Fine | 0.999646 | >99% | >0.999 |

Target: PCC ≥ 0.95 — **All stages exceed target by significant margin.**

## Throughput Benchmarks

> **Note:** Fill in measured values after running `run_bark_e2e.py` on N300.

| Stage | Metric | Target | Measured | Status |
| :--- | :--- | :--- | :--- | :--- |
| Semantic | Tokens/sec | ≥ 20 | _pending_ | — |
| Coarse | Tokens/sec | ≥ 60 | _pending_ | — |
| Fine | Time (s) | — | _pending_ | — |
| Decode | Time (s) | — | _pending_ | — |
| Overall | RTF | < 0.8 | _pending_ | — |
| Stretched | RTF | < 0.4 | _pending_ | — |

## Memory Budget

### Per-Stage Memory Breakdown

| Component | Location | Size (est.) | Notes |
| :--- | :--- | :--- | :--- |
| Embedding weights | DRAM | ~15 MB/stage | Shared for all codebooks in fine |
| Linear weights (QKV, MLP) | DRAM | ~45 MB/stage | 12 layers × (3H² + 2×4H²) |
| LayerNorm weights | DRAM | <1 MB/stage | 12 layers × 2 × (weight + bias) |
| KV cache (semantic) | DRAM | ~2 MB peak | Grows with seq len, deallocated after |
| KV cache (coarse) | DRAM | ~2 MB peak | Same strategy |
| Activations (per layer) | L1 | ~0.5 MB | Hidden states, QKV, attn scores |
| SDPA workspace | L1 | ~1 MB | Chunked (128×128) for long sequences |

### Memory Lifecycle

```
Text Input
    │
    ▼
┌────────────────────────────┐
│ Stage 1: Semantic          │
│ KV Cache: DRAM (growing)   │──→ Deallocated after generation
│ Activations: L1 (streamed) │
└────────────────────────────┘
    │ semantic_tokens (host)
    ▼
┌────────────────────────────┐
│ Stage 2: Coarse            │
│ KV Cache: DRAM (growing)   │──→ Deallocated after generation
│ Activations: L1 (streamed) │
└────────────────────────────┘
    │ coarse_tokens (host)
    ▼
┌────────────────────────────┐
│ Stage 3: Fine              │
│ No KV cache (non-causal)   │
│ Activations: L1            │
│ Codebooks: device tensors  │
└────────────────────────────┘
    │ fine_tokens (host)
    ▼
┌────────────────────────────┐
│ Stage 4: EnCodec (CPU)     │
│ Quantizer + Decoder        │
└────────────────────────────┘
    │ audio (numpy)
    ▼
  .wav file
```

## Sharding Strategy

### Stage 1 & 2 (Causal — Semantic, Coarse)

| Operation | Sharding | Memory | Rationale |
| :--- | :--- | :--- | :--- |
| Embedding | None (DRAM → L1) | ROW_MAJOR on device | ttnn.embedding requires 2D ROW_MAJOR |
| QKV Projection | Width-parallel | L1 | `ttnn.linear` auto-shards across cores |
| Attention (prefill) | Block sharded | L1 | `ttnn.transformer.scaled_dot_product_attention` with 128×128 chunks |
| Attention (decode) | Replicated | DRAM→L1 | Manual matmul Q×K^T, K^T deallocated immediately |
| KV Cache | Not sharded | DRAM | Grows linearly, too dynamic for L1 sharding |
| MLP (in_proj) | Width-parallel | L1 | 768→3072 expansion |
| MLP (out_proj) | Width-parallel | L1 | 3072→768 reduction |
| LayerNorm | Replicated | L1 | Small enough for single-core |
| LM Head | Width-parallel | DRAM→L1 | Large output (768→10048) may overflow L1 |

### Stage 3 (Non-Causal — Fine)

| Operation | Sharding | Memory | Rationale |
| :--- | :--- | :--- | :--- |
| Embedding (×8) | None | DRAM→L1 | One per codebook, ROW_MAJOR |
| Attention | Block sharded | L1 | Non-causal — no mask, full parallelism |
| MLP | Width-parallel | L1 | Same as stages 1-2 |
| LM Head (×6) | Width-parallel | L1 | One per predicted codebook (768→1024) |

## Inter-Stage Data Transfer Audit

| Transfer | Direction | Format | Optimization |
| :--- | :--- | :--- | :--- |
| Text → Semantic input | Host→Device | uint32 ROW_MAJOR | Minimal (tokenizer on CPU) |
| Semantic logits → argmax | Device→Host | bfloat16 TILE→float32 | Required for vocab masking |
| Semantic tokens → Coarse input | Host→Device | uint32 ROW_MAJOR | Tokens stay on host (small) |
| Coarse logits → argmax | Device→Host | bfloat16 TILE→float32 | Required for codebook masking |
| Coarse tokens → Fine input | Host→Device | uint32 ROW_MAJOR | Small tensor, unavoidable |
| Fine logits → argmax | On-device | ttnn.argmax | No transfer needed |
| Fine codebooks → EnCodec | Device→Host | int32 | Required (EnCodec is CPU) |

**Summary:** Host-device transfers during generation are minimal and intentional.
The only per-step transfer is logits→host for argmax (stages 1-2), which is
required for Bark's vocab masking strategy. Stage 3 uses on-device argmax.

## Pipeline Overlap Analysis

| Strategy | Est. Time | Speedup | Feasibility |
| :--- | :--- | :--- | :--- |
| Sequential (current) | T₁ + T₂ + T₃ + T₄ | 1.0× | Current implementation |
| Stage 1 ∥ Stage 2 | max(T₁,T₂) + T₃ + T₄ | ~1.3× | Requires 2 devices or async |
| Fully pipelined | max(T₁,T₂,T₃,T₄) | ~2.0× | Theoretical, needs 4 devices |

Current single-device implementation is sequential. Multi-device overlap is
documented in `bark_pipeline_overlap.py` with latency estimates.

## Optimizations Applied

### Stage 2 Optimizations
- [x] DRAM weights, L1 activations (optimal memory hierarchy)
- [x] Width-sharded linear ops (auto by ttnn.linear)
- [x] Block-sharded SDPA with 128×128 chunking
- [x] KV cache in DRAM (avoids L1 overflow during long sequences)
- [x] Fused LayerNorm (ttnn.layer_norm, not manual)
- [x] On-device GELU_NEW decomposed activation
- [x] Explicit intermediate tensor deallocation
- [x] Causal mask generated once per prefill, not per decode step

### Stage 3 Optimizations
- [x] 56-core compute grid (8×7 on N300)
- [x] HiFi4 math fidelity with fp32 accumulation
- [x] On-device argmax for fine stage codebook prediction
- [x] Transposed key tensor immediate deallocation (L1 pressure fix)
- [x] KV cache explicit deallocation after generation completes
- [x] Pipeline overlap analysis and estimation module
- [x] Long text support (500+ chars via sentence chunking + crossfade)
- [x] Voice preset loading with in-memory caching
- [x] Batch processing with per-item RTF metrics
- [x] Comprehensive memory budget documentation

## Known Limitations

1. **Batch size:** Currently limited to batch=1 for autoregressive generation
2. **Sampling:** Greedy argmax only (no top-k/top-p/temperature)
3. **Host-side argmax:** Logits transfer to host each decode step for vocab masking (stages 1-2)
4. **Pipeline overlap:** Sequential execution on single device (multi-device overlap documented)
5. **EnCodec:** CPU-only decode (not ported to TTNN)
6. **Long sequences:** KV cache may hit DRAM limits for very long generations (>2048 tokens)

## Test Commands

```bash
# Unit tests (requires N300)
pytest models/demos/wormhole/bark/tests/test_bark_model.py -svv

# CPU-only parity tests
pytest models/demos/wormhole/bark/tests/test_bark_reference_parity.py -svv

# End-to-end pipeline (requires N300)
python models/demos/wormhole/bark/tests/run_bark_e2e.py

# Token accuracy validation (requires N300)
python models/demos/wormhole/bark/tests/validate_token_accuracy.py

# Per-stage profiling (requires N300)
python models/demos/wormhole/bark/tests/profile_bark.py
```
