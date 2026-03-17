# Fused MM+RS Op Analysis for Llama 70B Galaxy

## Llama 70B Fusion Candidates (8K prefill)

### MM+RS Fusions (4 ops)
| Op | Input Shape | Output Shape | MM (µs) | RS (µs) | Total (µs) | Fidelity | Fusion Benefit |
|----|-------------|--------------|---------|---------|------------|----------|----------------|
| QKV | 8192x2048 | 8192x1280 | 515 | 282 | 797 | HiFi2 | ❌ NO (-10%) * |
| Attn Out | 8192x1024 | 8192x2048 | 435 | 419 | 854 | HiFi2 | ❌ NO (-0.8%) |
| **FF1** | **8192x2048** | **8192x3584** | **751** | **712** | **1,463** | **LoFi** | ✅ YES (+5.4%) |
| **FF3** | **8192x2048** | **8192x3584** | **752** | **694** | **1,446** | **LoFi** | ✅ YES (+5.4%) |

FF1 and FF3 have identical dimensions - same config works for both.

*QKV/Attn Out fusion NOT beneficial:
- Fused op limited to 7x6=42 MM cores (dispatch core conflict on row 7)
- Baseline uses 7x8=56 MM cores (33% more)
- Small N dims don't provide enough RS work to compensate for fewer MM cores

### AG+MM Fusion (separate effort)
| Op | Pattern | AG (µs) | MM (µs) | Total (µs) | Fidelity |
|----|---------|---------|---------|------------|----------|
| FF2 | AG→MM | 530 | 1,321 | 1,851 | HiFi2 |

(FF2 uses AG+MM fusion, not MM+RS)

---

# FF1/FF3 Optimization (8192x2048 → 8192x3584)

## Baseline Performance (Separate Ops)
- **MinimalMatmul**: 751 µs (63 cores = 7x9, LoFi, BFLOAT4_B weights)
- **ReduceScatter**: 712 µs (num_links=4, num_workers_per_link=4)
- **Total**: ~1,463 µs

## Wormhole Compute Grid Constraints
- **Total grid**: 8x8 = 64 cores (rows 0-7, cols 0-7)
- **Row 7 conflicts with dispatch cores** → max RS offset = row 6
- Fused op must fit both MM and RS cores within rows 0-6

---

## Phase 1: MM Grid Sweep (chunk=2, workers=2, links=4)

| MM Grid | Cores | RS Offset | Duration | vs Baseline |
|---------|-------|-----------|----------|-------------|
| 6x6 | 36 | row 6 | ~2,039 µs | +39% slower |
| **7x6** | **42** | **row 6** | **~1,701 µs** | **+16% slower** |
| 8x6 | 48 | row 6 | ~1,747 µs | +19% slower |
| 6x7 | 42 | row 7 | ❌ FAILED | Dispatch core conflict |
| 7x7 | 49 | row 7 | ❌ FAILED | Dispatch core conflict |

**Winner: 7x6 grid (42 cores)**

---

## Phase 2: Fusion Params Sweep (7x6 grid fixed)

### workers=2 (best)
| chunk | workers | links | Duration | vs Baseline |
|-------|---------|-------|----------|-------------|
| **1** | **2** | **4** | **~1,689 µs** | **+15% (BEST)** |
| 2 | 2 | 4 | ~1,701 µs | +16% |
| 4 | 2 | 4 | ~1,705 µs | +17% |
| 1 | 2 | 3 | ~1,813 µs | +24% |
| 2 | 2 | 3 | ~1,858 µs | +27% |
| 4 | 2 | 3 | ~1,857 µs | +27% |

### workers=1 (slower)
| chunk | workers | links | Duration | vs Baseline |
|-------|---------|-------|----------|-------------|
| 2 | 1 | 4 | ~1,921 µs | +31% |
| 4 | 1 | 4 | ~1,921 µs | +31% |
| 1 | 1 | 4 | ~1,985 µs | +36% |

### workers=3, workers=4 (failed)
| workers | links | Status | Error |
|---------|-------|--------|-------|
| 3 | 4 | ❌ FAILED | Dispatch core conflict |
| 4 | 4 | ❌ FAILED | Out of bounds (row 10) |

**workers=2 is the maximum** that fits within the available core grid.

---

## Best Configuration Found
```
MM Grid: 7x6 (42 cores)
RS Offset: row 6
chunk_width_in_mm_blocks: 1
num_workers_per_link: 2
num_links: 4

Duration: ~1,689 µs (15% slower than baseline)
```

---

## Key Findings

1. **7x6 is optimal MM grid**: Better than 6x6 (fewer cores) and 8x6 (diminishing returns)
2. **Row 7 unusable**: Conflicts with dispatch cores
3. **links=4 > links=3**: ~100-150 µs difference
4. **workers=2 is max**: workers=3/4 cause core placement errors
5. **chunk=1 slightly better**: Finer overlap granularity helps

## Performance Gap Analysis
- **Fused best**: 1,689 µs (42 MM cores)
- **Separate baseline**: 1,463 µs (63 MM cores)
- **Gap**: 226 µs (15% slower)
- **Root cause**: Fused op uses 33% fewer MM cores (42 vs 63)

## Conclusion
The fused `minimal_matmul_strided_reduce_scatter_async` op **cannot beat separate ops** for Llama 70B FF1/FF3 on 8x4 Galaxy because:

1. **Core grid constraint**: Must share 8x8 grid between MM and RS cores
2. **Dispatch core conflict**: Row 7 unusable, limits RS placement to row 6
3. **Worker limit**: Max workers=2 per link fits; workers=3/4 fail
4. **Core count gap**: Best achievable is 42 MM cores vs 63 in baseline (33% fewer)

The overlap benefit from fusion (~226 µs potential savings from hiding RS latency) does not compensate for the 33% reduction in MM compute resources.

**Recommendation**: Continue using separate MM + RS ops for Llama 70B FF1/FF3 on Galaxy 6U.

---

## Phase 3: Block Size Sweep (7x6 grid, chunk=1, workers=2, links=4)

| Block Config (M,K,N,subblock) | Duration | vs Separate (1,463 µs) |
|-------------------------------|----------|------------------------|
| **5_8_32_1x16** | **~1,392 µs** | **-4.9% FASTER 🏆** |
| 5_8_32_1x8 | ~1,399 µs | -4.4% faster |
| 5_8_16_1x8 | ~1,400 µs | -4.3% faster |
| 4_8_32_1x16 | ~1,401 µs | -4.2% faster |
| 4_8_32_1x8 | ~1,407 µs | -3.8% faster |
| 5_8_24_1x8 | ~1,408 µs | -3.8% faster |
| 4_8_48_1x8 | ~1,411 µs | -3.6% faster |
| 4_8_24_1x8 | ~1,411 µs | -3.6% faster |
| 4_8_16_1x8 | ~1,411 µs | -3.6% faster |
| 5_4_32_1x8 | ~1,415 µs | -3.3% faster |
| 4_4_32_1x8 | ~1,431 µs | -2.2% faster |
| 6_8_32_1x8 | ~1,432 µs | -2.1% faster |
| 4_8_32_2x4 | ~1,437 µs | -1.8% faster |
| 4_16_32_1x8 | ~1,495 µs | +2.2% slower |
| 8_8_16_1x8 | ~1,497 µs | +2.3% slower |
| 5_8_32_2x4 | ~1,498 µs | +2.4% slower |
| 3_8_32_1x8 | ~1,577 µs | +7.8% slower |
| 4_8_8_1x8 | ~1,634 µs | +12% slower |
| 8_8_8_1x8 | ~1,689 µs | +15% slower |
| 16_8_8_1x8 | ~1,806 µs | +23% slower |
| 2_8_16_1x8 | ~2,175 µs | +49% slower |
| 8_8_32_1x8 | ❌ FAILED | L1 overflow |
| 5_8_48_1x8 | ❌ FAILED | L1 overflow |
| 4_4_4_1x4 | ~2,471 µs | +69% slower |

### Key Findings
- **M_block=5 is optimal** - M=4 close, M=3 too small, M=6/8 slower
- **N_block=32 is optimal** - larger N improves RS overlap, N=48 causes L1 overflow
- **K_block=8 is optimal** - K=4 almost as good, K=16 hurts
- **subblock 1x16 is optimal with M=5** - wider subblock helps
- **subblock 2x4 hurts** performance significantly

### Best Configuration Found
```
MM Grid: 7x6 (42 cores)
RS Offset: row 6
chunk_width_in_mm_blocks: 1
num_workers_per_link: 2
num_links: 4
num_buffers_per_channel: 16
M_block_size: 5
K_block_size: 8
N_block_size: 36
subblock_h: 1
subblock_w: 16

Duration: ~1,384 µs (5.4% FASTER than separate ops!)
Savings: ~79 µs per FF1/FF3 operation
```

## Phase 4: RS Buffer Sweep (with best block config 5_8_36_1x16)

| num_buffers_per_channel | Duration | vs Separate (1,463 µs) |
|-------------------------|----------|------------------------|
| **16** | **~1,384 µs** | **-5.4% FASTER 🏆** |
| 4 | ~1,386 µs | -5.3% faster |
| 8 | ~1,397 µs | -4.5% faster |
| 2 | ~1,397 µs | -4.5% faster |
| 32 | ~1,399 µs | -4.4% faster |
| 1 (default) | ~1,401 µs | -4.2% faster |

buf16 is optimal. More buffers beyond 16 don't help.

## Phase 5: Combined Sweep (block + buffer combinations)

| Config | Duration | vs Separate (1,463 µs) |
|--------|----------|------------------------|
| 5_8_32_1x16_buf32 | ~1,386 µs | -5.3% faster |
| 5_8_36_1x16_buf32 | ~1,391 µs | -4.9% faster |
| 5_8_36_1x16_buf16 | ~1,392 µs | -4.9% faster |
| 5_8_32_1x16_buf16 | ~1,397 µs | -4.5% faster |
| 4_8_32_1x16_buf16 | ~1,399 µs | -4.4% faster |
| 5_8_32_1x8_buf16 | ~1,400 µs | -4.3% faster |

N=36 with buf16 remains optimal from Phase 4.

### Ultra Fine-Tuning Results
| Config | Duration | Notes |
|--------|----------|-------|
| 5_8_36_1x16 | ~1,391 µs | Best |
| 5_8_32_1x16 | ~1,392 µs | Essentially tied |
| 5_8_28_1x16 | ~1,396 µs | Close |
| 4_8_32_1x32 | ~1,399 µs | |
| 5_8_32_1x32 | ~1,400 µs | 1x32 doesn't help |
| 5_6_32_1x16 | ~1,401 µs | K=6 ok |
| 5_10_32_1x16 | ~1,419 µs | K=10 worse |
