# OLMo-3.1-32B Status

**Last Updated**: 2026-03-11

## Quick Summary

| Mode | Status | Performance |
|------|--------|-------------|
| Prefill | ✅ Working | TTFT: 400 ms @ 4k, 51 ms @ 128 (traced, 64 layers) |
| Decode | ✅ Working | 10 iterations pass, hybrid CCL approach |
| Tracing (Prefill) | ✅ Working | 1.77-6.38x speedup |
| Tracing (Decode) | ❌ Not Done | Pending |

---

## Prefill Mode ✅

**Status**: Fully functional with tracing enabled

| Component | Status | PCC |
|-----------|--------|-----|
| RMSNorm | ✅ | 0.9999 |
| SwiGLU MLP | ✅ | 0.9995 |
| YaRN RoPE | ✅ | 0.99999 |
| GQA Attention | ✅ | 0.9626 |
| Sliding Window SDPA | ✅ | >0.98 |
| Full 64-layer forward | ✅ | No NaN/Inf |
| Tracing | ✅ | 1.77-6.38x speedup |

**Performance (64 layers, batch=1)**:
| Seq Length | Non-trace | Trace | Speedup |
|------------|-----------|-------|---------|
| 128 | 323 ms | **51 ms** | **6.38x** |
| 4096 | 707 ms | **400 ms** | **1.77x** |

**Test Commands**:
```bash
export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think

# Prefill test (64 layers, 4k seq)
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_decoder_prefill.py -v -x -k "64layers and 4k"

# Prefill trace test with performance measurement
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_decoder_prefill.py::test_olmo_prefill_trace -v -k "64layers"
```

---

## Decode Mode ✅

**Status**: Functional with hybrid CCL (partial device-side)

| Component | Status | Notes |
|-----------|--------|-------|
| Q head padding (5→8) | ✅ | For fused RoPE compatibility |
| K head expansion | ✅ | 1→8 heads for RoPE, slice back to 1 |
| Sliding window decode | ✅ | 4096 window size |
| MLP forward | ✅ | Fixed weight dimension mismatch |
| 10-iteration test | ✅ | mean=0.0022, std=0.58 |

**Test Command**:
```bash
export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
pytest models/demos/llama3_70b_galaxy/tests/test_olmo_decode.py::test_olmo_decoder_decode -v -x
```

---

## CCL Operations Status

### Device-side ✅ (3/5 operations)

| Operation | Location | Memory Config |
|-----------|----------|---------------|
| MLP FF1/FF3 all_gather | `llama_mlp.py:306` | BINARY_MUL buffer (3840 width) |
| MLP W2 all_reduce | `llama_mlp.py:369` | FF2_OUT_RING_MEMCFG_OLMO |
| Attention WO all_reduce | `llama_attention.py:828` | SHARDED_WO_OUT_RING_MEMCFG_OLMO |

### Host-side ❌ (2/5 operations - blocked)

| Operation | Location | Blocker |
|-----------|----------|---------|
| MLP W1/W3 reduce_scatter | `llama_mlp.py:185` | Kernel expects matching input/output shard counts |
| Attention post-SDPA all_gather | `llama_attention.py:736` | `all_gather_concat` crashes for OLMo dims |

**Why blocked**: OLMo's dimensions don't fit existing kernel constraints:
- 5:1 GQA ratio (Llama/Qwen use 8:1)
- 3456 intermediate per TP (Llama: 3584, Qwen: 3200)
- 1280 dim per TP (Llama/Qwen: 2048)

---

## Recent Fixes

### Prefill Tracing (2026-03-11)
1. **llama_ccl.py**: Fixed `ring_reduce_scatter` buffer selection
   - Pass only output buffer to `reduce_scatter_minimal_async`
   - Was incorrectly passing `[intermediate, output]` causing shape mismatch

2. **llama_decoder.py**: Added `enable_trace` flag
   - Skip input tensor deallocation during trace capture
   - Required because trace replay needs persistent input buffers

3. **llama_attention.py**: Created unpadded WO weight for prefill
   - Decode uses padded WO (5→8 heads for fused RoPE)
   - Prefill SDPA output is unpadded, needs matching WO weight

---

## TODO

### High Priority
- [ ] **Enable tracing for decode** - Apply similar `enable_trace` pattern
- [ ] **End-to-end demo validation** - Run `demo_olmo_decode.py` with full model
- [ ] **Performance benchmark** - Measure decode tok/s

### Medium Priority
- [ ] **Device-side reduce_scatter** - Requires kernel modification to handle different input/output shard counts
- [ ] **Device-side all_gather_concat** - Requires kernel fix for OLMo's batch size (32 vs 128)

### Low Priority
- [ ] **Code cleanup** - Remove DEBUG print statements
- [ ] **Test cleanup** - Replace prints with proper assertions

---

## Key Files

| File | Purpose |
|------|---------|
| `tt/olmo_model_config.py` | OLMo-specific model configuration |
| `tt/llama_attention.py` | Attention with Q/K head padding for RoPE |
| `tt/llama_mlp.py` | MLP with hybrid CCL path |
| `tt/llama_ccl.py` | CCL operations with OLMo buffers |
| `tests/test_olmo_decode.py` | Decode mode tests |
| `tests/test_olmo_decoder_prefill.py` | Prefill mode tests |
| `demo/demo_olmo_decode.py` | End-to-end demo (needs validation) |

---

## OLMo vs Llama/Qwen Differences

| Aspect | Llama3-70B | Qwen3-32B | OLMo-3.1-32B |
|--------|------------|-----------|--------------|
| Q Heads | 64 | 64 | **40** |
| KV Heads | 8 | 8 | **8** |
| GQA Ratio | 8:1 | 8:1 | **5:1** |
| Intermediate | 28672 | 25600 | **27648** |
| dim_per_tp | 2048 | 2048 | **1280** |
| RoPE Type | Linear | Linear | **YaRN** |
| Sliding Window | None | None | **4096** |

These differences cause kernel compatibility issues with CCL operations designed for Llama/Qwen dimensions.
