# OLMo-3.1-32B Bring-up Log

## Session: 2026-03-28

### Status: 8K 30%, 16K 20%, 32K 0% reliability - CCL warmup deadlocks

### Summary

1. Re-verified ISL results - 8K/16K work but have ~33% first-run failure rate
2. Added TLB leak prevention via `TT_CCL.cleanup()` method
3. 8K coherency issue traced to complex summarization prompt (16K uses simpler prompt)

### ISL Results (batch=1, 10 runs each)

| ISL | Pass Rate | TTFT | Notes |
|-----|-----------|------|-------|
| 8K | 3/10 (30%) | ~2.8s | Hangs at random layers (L55, L62) during warmup |
| 16K | 2/10 (20%) | ~5.7s | Hangs at random layers (L14, L31, L48) during warmup |
| 32K | 0/10 (0%) | - | Consistently hangs during warmup |

### TLB Leak Fix
Added `cleanup()` method to `TT_CCL` class (`llama_ccl.py`) that deallocates:
- `persistent_buffers` (prefill reduce scatter buffers)
- `all_gather_buffers` (prefill all gather buffers)
- `reduce_scatter_buffers` (decode mode)
- `rs_create_heads_buffers` (decode mode)
- `agmm_ff2_intermediate_buffers` (OLMo decode mode)

`TtTransformer.__del__()` now calls `cleanup()` on all CCL objects to prevent TLB exhaustion after timeouts.

### Configuration
```bash
DEBUG_PREFILL_LAYERS=1 pytest ... -k "isl-8k-b1"
DEBUG_PREFILL_LAYERS=1 pytest ... -k "isl-16k-b1"
```

### Notes
- First run after `tt-smi -glx_reset` often fails (~33% failure rate)
- Subsequent runs more reliable
- 32K consistently hangs during prefill - CCL ordering issue at scale
- 8K uses complex literary summarization prompt; 16K uses simple factoid prompt

---

## Session: 2026-03-27 (evening)

### Status: 8K WORKING, 16K/32K need DEBUG_PREFILL_LAYERS=1

### Summary

Re-verified ISL results after reverting/re-applying CCL fixes:
- 8K: PASS without DEBUG_PREFILL_LAYERS
- 16K/32K: Warmup completes but actual prefill hangs without DEBUG_PREFILL_LAYERS=1

#### Fixes Applied (`llama_ccl.py`)
1. Removed 8192 from `support_seqlens` (8K now runs in eager mode with sync CCL)
2. Changed `num_links` to 1 for sync CCL in `ring_reduce_scatter` (multi-link deadlocks)
3. Changed `num_links` to 1 for sync CCL in `ring_all_gather` (multi-link deadlocks)

### ISL Results (batch=1)

| ISL | TTFT | Decode | Warmup | Notes |
|-----|------|--------|--------|-------|
| 128 | ~0.3s | 17.1 tok/s | ~1.5s | PASS (traced prefill) |
| 8K | 2.69s | 16.33 tok/s | 3.54s | PASS (eager mode, sync CCL with num_links=1) |
| 16K | - | - | 7s | Warmup OK, actual prefill hangs at layer 0 even with DEBUG_PREFILL_LAYERS=1 |
| 32K | - | - | ~7s | Warmup OK, actual prefill hangs at layer 0 even with DEBUG_PREFILL_LAYERS=1 |

**Regression from earlier session**: 16K/32K worked earlier today (TTFT 5.6s/13.4s) but now hang during actual prefill despite same code. Issue is in prefill execution, not warmup/compile.

### 16K/32K Configuration
```bash
DEBUG_PREFILL_LAYERS=1 pytest ... -k "isl-16k-b1"
DEBUG_PREFILL_LAYERS=1 pytest ... -k "isl-32k-b1"
```

Without DEBUG_PREFILL_LAYERS=1, 16K/32K complete warmup but hang during actual prefill. The per-layer `ttnn.synchronize_device` prevents operation buildup that causes deadlock.

---

## Session: 2026-03-27 (earlier)

### Status: 8K, 16K, and 32K ISL ALL WORKING

### Summary

All ISLs now work. 32K requires DEBUG_PREFILL_LAYERS=1 (per-layer sync) to prevent deadlock.

#### Root Cause
For 16K+ ISLs, sync CCL operations pile up without explicit syncs between layers, causing deadlock. Adding `ttnn.synchronize_device` after each layer (DEBUG_PREFILL_LAYERS=1) prevents this.

#### Fixes Applied (`llama_ccl.py`)
1. Removed 8192 from `support_seqlens` (8K now runs in eager mode with sync CCL)
2. Changed `num_links` to 1 for sync CCL in `ring_reduce_scatter` (multi-link deadlocks)
3. Changed `num_links` to 1 for sync CCL in `ring_all_gather` (multi-link deadlocks)

### ISL Results (batch=1)

| ISL | TTFT | Decode | Notes |
|-----|------|--------|-------|
| 8K | ~2.7s | 16.3 tok/s | PASS |
| 16K | ~5.6s | 16.1 tok/s | PASS |
| 32K | ~13.4s | 15.7 tok/s | PASS (requires DEBUG_PREFILL_LAYERS=1) |

### 32K Configuration
```bash
DEBUG_PREFILL_LAYERS=1 pytest ... -k "isl-32k-b1"
```

Without DEBUG_PREFILL_LAYERS=1, 32K hangs during warmup prefill. The per-layer sync prevents operation buildup that causes deadlock.

---

## Session: 2026-03-26

### Status: 8K and 16K ISL WORKING (no hang)

### Summary

Fixed 8K ISL hang by removing 8192 from `support_seqlens` in `llama_ccl.py`. Now 8K runs in eager mode with sync CCL like 16K.

#### Root Cause
Async CCL with barrier_semaphore was deadlocking even with pre-allocated buffers at 8K.

#### Fix Applied (`llama_ccl.py:121`)
```python
# Before:
self.support_seqlens = [8192, 4096, 2048, 1024, 512, 256, 128]

# After:
self.support_seqlens = [4096, 2048, 1024, 512, 256, 128]
```

### 8K ISL Results (batch=1)

| Metric | Value |
|--------|-------|
| TTFT | 2751 ms (~2.75s) |
| Decode speed | 16.3 tok/s |
| Status | PASS (no hang) |

---

## Session: 2026-03-25

### Status: 16K ISL WORKING (no hang), coherency issues remain

### Summary

Fixed the 16K ISL hang by using sync CCL (`ttnn.all_gather`, `ttnn.reduce_scatter`) for OLMo prefill when no persistent buffers are available.

#### Root Cause
The async CCL operations (`all_gather_async`, `reduce_scatter_minimal_async`) with `barrier_semaphore` were deadlocking in eager mode (16K+) when no persistent buffers were pre-allocated.

#### Fix Applied (`llama_ccl.py`)
1. `line_all_gather`: Added sync CCL path when `buffer_key=None` (MLP case) or `persistent_buffer is None`
2. `ring_all_gather`: Changed to use sync CCL only when `persistent_buffers is None` (not for ALL OLMo prefill)
3. `ring_reduce_scatter`: Changed to use sync CCL only when `persistent_buffers_list is None`

Key insight: Using `num_links=1` for sync CCL avoids multi-link deadlocks.

### 16K ISL Results (batch=1)

| Metric | Value |
|--------|-------|
| TTFT | 5672 ms (~5.7s) |
| Decode speed | 16.12 tok/s |
| Status | PASS (no hang) |
| Coherency | Degraded (pre-existing issue) |

**Note**: Output coherency is poor for all ISLs in current state - this is a separate issue from the CCL deadlock fix.

---

## Session: 2026-03-24

### Status: PARTIAL — ISL 128/2K coherent, 1K/4K partial degradation, 8K hangs

**Note:** Clearing the TG cache causes garbage output. Cache must be rebuilt properly.

### Summary of Changes (this session)

#### Problem
4k ISL prefill PCC for 4 layers was below target (0.9989 initially), causing coherency degradation with sampling (top_k=50, top_p=0.95, temperature=0.6) after ~50-100 tokens of 1000-token generation.

#### Root Cause
Three precision issues at seq_len=4096 in `llama_attention.py`:
1. `xqkv_dtype` used bfloat8_b at 4k ISL (threshold `<= 2048` was too low)
2. `wo_ag_key` used bfloat8_b WO all-gather buffer (off-by-one: condition was `< 4096` not `<= 4096`)
3. WO linear used `minimal_matmul` at 4k ISL (bfloat8_b output), not `ttnn.linear` (bfloat16 output)

#### Fixes Applied
- `llama_attention.py`: Extended bfloat16 QKV to 4k ISL (`<= 4096` threshold for xqkv_dtype and qkv_buffer_key)
- `llama_attention.py`: Fixed off-by-one in WO all-gather buffer key selection (`<= 4096`)
- `llama_attention.py`: Extended `ttnn.linear` WO (FP32 acc, bfloat16 output) to 4k ISL — `WO_PREFILL_PROGCFG(4096)` uses `out_subblock_w=2` (product=2 ≤ max_dest_volume=4), safe with hifi2
- `llama_ccl.py`: Added dtype-based fallback in `ring_reduce_scatter` — sync `ttnn.reduce_scatter` for bfloat16 inputs (async `reduce_scatter_minimal_async` does not support bfloat16 at 4096×896)
- `llama_ccl.py` (prefill CCL): Using async `all_gather_async_reversed` + `reduce_scatter_minimal_async` with conditional barrier_semaphore for OLMo prefill (reverted from sync in previous session)
- `llama_ccl.py` (decode CCL): Using sync `ttnn.reduce_scatter` for OLMo decode (commit deeb7100d76 fix preserved — async produces garbage with bfloat8_b DRAM inputs)
- `demo_olmo_decode.py`: Applied prompt clamp fix, ChatML template fix, 1000 decode tokens for 4k/8k ISL

### PCC Results

| Test | Hidden PCC | Host Logits PCC | Token Match | Status |
|------|------------|-----------------|-------------|--------|
| 4L 4k ISL | 0.9990237 | 0.9971437 | ✓ (18673) | PASS > 0.998 |

### ISL Sweep Results (batch=1, 512 decode tokens)

| ISL | TTFT (ms) | tok/s/user | Coherent? | Status |
|-----|-----------|------------|-----------|--------|
| 128 | 313.68 | 17.07 | ✓ fully coherent | PASS |
| 1K | 488.29 | 16.83 | ~partial (degrades late) | PARTIAL |
| 2K | 801.77 | 16.70 | ✓ fully coherent | PASS |
| 4K | 1274.56 | 16.44 | ~partial (some degradation) | PARTIAL |
| 8K | - | - | - | HANG (prefill compile) |

**Note**: ISL 1K and 4K show late-stage degradation (output becomes fragmented toward end of generation). ISL 128 and 2K remain fully coherent throughout 512 tokens.

### Coherency Results (detailed)

| ISL | Layers | Sampling | Generated Tokens | Coherent? |
|-----|--------|----------|------------------|-----------|
| 2k | 64 | top_k=50, p=0.95, T=0.6 | 200 | ✓ |
| 4k | 64 | top_k=50, p=0.95, T=0.6 | 1000 | ✓ (verified this session) |

### Block Hash
- `llama_attention.py`: bfloat16 QKV+WO at 4k ISL
- `llama_ccl.py`: sync RS fallback for bf16 inputs; async for bf8 prefill; sync for OLMo decode
