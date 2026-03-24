# OLMo-3.1-32B Bring-up Log

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
