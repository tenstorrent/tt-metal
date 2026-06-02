# Flux2 WH Galaxy — Matmul Block Size Sweep Results

**Date:** 2026-06-01
**Branch:** `flux2_wh_glxy`
**Device:** WH Galaxy 4x8 (32 devices, Topology.Ring)
**Model config:** sp=4, tp=8, image 1024x1024, 50 denoising steps

---

## Baseline (Single Transformer Block Profile)

Profiled from: `generated/profiler/reports/2026_05_31_20_23_24/ops_perf_results_2026_05_31_20_23_24.csv`

| Op Type | Invocations | Mean (us) | Total (ms) | % of Total |
|---------|-------------|-----------|------------|------------|
| Plain MM (MinimalMatmul) | 320 | 330 | 158.5 | 11.2% |
| AGMM (AllGather+Matmul) | 256 | 761 | 291.0 | 20.5% |
| MM+RS (Matmul+ReduceScatter) | 128 | 1121 | 213.7 | 15.1% |
| Other (SDPA, RMSNorm, etc.) | — | — | 756.1 | 53.3% |
| **Total** | — | — | **1419.4** | 100% |

---

## Sweep Results — Plain MM (8x9 core grid)

Sweep CSV: `sweep_results_mm.csv`

### Best Configs (Winner per Shape)

| Shape (M, K, N) | Use Case | M_blk | K_blk | N_blk | Subblock (h,w) | Time (us) | Default (us) | Improvement |
|-----------------|----------|-------|-------|-------|----------------|-----------|--------------|-------------|
| (1024, 6144, 2304) | ff1 | 4 | 6 | 10 | (2, 2) | **451** | ~870 | ~48% |
| (512, 6144, 2304) | ff1 | 2 | 8 | 5 | (2, 1) | **392** | ~710 | ~45% |
| (512, 15360, 768) | context_embedder | 3 | 10 | 3 | (1, 3) | **357** | ~375 | ~5% |
| (1024, 6144, 128) | proj_out | 2 | 12 | 1 | (2, 1) | **170** | ~305 | ~44% |

### Top 5 per Shape

**(1024, 6144, 2304) — ff1:**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 4 | 6 | 10 | (2, 2) | 451 |
| 2 | 6 | 6 | 10 | (2, 2) | 453 |
| 3 | 4 | 6 | 18 | (2, 2) | 455 |
| 4 | 8 | 6 | 10 | (2, 2) | 455 |
| 5 | 4 | 8 | 10 | (2, 2) | 457 |

**(512, 6144, 2304) — ff1:**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 2 | 8 | 5 | (2, 1) | 392 |
| 2 | 4 | 8 | 5 | (4, 1) | 393 |
| 3 | 3 | 8 | 5 | (3, 1) | 393 |
| 4 | 3 | 6 | 10 | (3, 1) | 394 |
| 5 | 2 | 6 | 9 | (1, 3) | 394 |

**(512, 15360, 768) — context_embedder:**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 3 | 10 | 3 | (1, 3) | 357 |
| 2 | 4 | 8 | 3 | (4, 1) | 357 |
| 3 | 3 | 10 | 4 | (1, 4) | 358 |
| 4 | 5 | 10 | 3 | (1, 3) | 358 |
| 5 | 3 | 10 | 6 | (1, 3) | 358 |

**(1024, 6144, 128) — proj_out:**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 2 | 12 | 1 | (2, 1) | 170 |
| 2 | 2 | 12 | 2 | (2, 2) | 171 |
| 3 | 2 | 16 | 1 | (2, 1) | 171 |
| 4 | 2 | 8 | 1 | (2, 1) | 171 |
| 5 | 4 | 8 | 1 | (4, 1) | 171 |

---

## Sweep Results — AGMM (8x8 core grid)

Note: AGMM K_tiles are per-device (K_total / tp_size). For WH tp=8: K=768 → K_tiles=3.
Trace capture is disabled for AGMM on WH (Event Synchronization not supported during trace).

### Best Configs (Winner per Shape)

| Shape (M, K, N) | Use Case | M_blk | K_blk | N_blk | Subblock (h,w) | Time (us) | Default (us) | Improvement |
|-----------------|----------|-------|-------|-------|----------------|-----------|--------------|-------------|
| (1024, 768, 4608) | qkv | 20 | 3 | 6 | (2, 2) | **355** | ~761 | ~53% |
| (512, 768, 4608) | qkv | 10 | 3 | 12 | (1, 4) | **242** | ~761 | ~68% |
| (1024, 768, 768) | to_out | 4 | 1 | 16 | (1, 4) | **190** | ~761 | ~75% |
| (512, 768, 768) | to_out | 2 | 1 | 16 | (1, 4) | **132** | ~761 | ~83% |

### Top 5 per Shape

**(1024, 768, 4608) — QKV projection (chunks=3):**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 20 | 3 | 6 | (2, 2) | 355 |
| 2 | 10 | 3 | 12 | (1, 4) | 361 |
| 3 | 4 | 3 | 6 | (2, 2) | 368 |
| 4 | 12 | 3 | 8 | (1, 4) | 368 |
| 5 | 8 | 3 | 6 | (2, 2) | 370 |

**(512, 768, 4608) — QKV projection (chunks=3):**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 10 | 3 | 12 | (1, 4) | 242 |
| 2 | 3 | 3 | 12 | (1, 4) | 252 |
| 3 | 2 | 3 | 16 | (1, 4) | 253 |
| 4 | 12 | 3 | 8 | (1, 4) | 254 |
| 5 | 2 | 3 | 12 | (1, 4) | 255 |

**(1024, 768, 768) — to_out:**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 4 | 1 | 16 | (1, 4) | 190 |
| 2 | 5 | 1 | 16 | (1, 4) | 192 |
| 3 | 6 | 1 | 16 | (1, 4) | 197 |
| 4 | 16 | 3 | 3 | (4, 1) | 202 |
| 5 | 20 | 3 | 3 | (4, 1) | 205 |

**(512, 768, 768) — to_out:**

| Rank | M_blk | K_blk | N_blk | sb (h,w) | Time (us) |
|------|-------|-------|-------|----------|-----------|
| 1 | 2 | 1 | 16 | (1, 4) | 132 |
| 2 | 4 | 1 | 16 | (1, 4) | 140 |
| 3 | 3 | 1 | 16 | (1, 4) | 140 |
| 4 | 2 | 3 | 16 | (1, 4) | 140 |
| 5 | 3 | 3 | 16 | (1, 4) | 145 |

---

## Sweep Results — Plain MM on 8x8 grid (Round 2, 2026-06-01)

These shapes were missing from round 1 — they appear in the full pipeline but not in the single transformer block profile.

### Best Configs

| Shape (M, K, N) | M_blk | K_blk | N_blk | Subblock (h,w) | Time (us) |
|-----------------|-------|-------|-------|----------------|-----------|
| (1024, 6144, 4608) | 4 | 8 | 10 | (2, 2) | **792** |
| (512, 6144, 4608) | 2 | 16 | 6 | (2, 2) | **718** |
| (1024, 6144, 768) | 4 | 4 | 4 | (1, 4) | **207** |
| (512, 6144, 768) | 2 | 8 | 5 | (2, 1) | **151** |

## Sweep Results — Plain MM on 8x9 grid (Round 2, 2026-06-01)

| Shape (M, K, N) | M_blk | K_blk | N_blk | Subblock (h,w) | Time (us) |
|-----------------|-------|-------|-------|----------------|-----------|
| (1024, 6144, 4608) | 4 | 8 | 10 | (2, 2) | **818** |
| (512, 6144, 4608) | 2 | 8 | 9 | (1, 3) | **750** |
| (1024, 128, 768) | 6 | 2 | 3 | (1, 3) | **20** |

---

## Remaining Sweeps (TODO)

| Shape (M, K, N) | Op Type | Grid | Status |
|-----------------|---------|------|--------|
| (1024, 2304, 6144) | MM+RS | 8x7 | Sweep script ready |
| (512, 2304, 6144) | MM+RS | 8x7 | Sweep script ready |
| (1024, 3072, 6144) | MM+RS | 8x7 | Sweep script ready |
| (512, 3072, 6144) | MM+RS | 8x7 | Sweep script ready |

---

## All Configs Added to matmul.py

### grid_89_configs (8x9 grid — plain MM)

```python
# (M, K, N): (M_block, K_block, N_block, (subblock_h, subblock_w))
(1024, 6144, 2304): (4, 6, 10, (2, 2)),  # 451 us
(512, 6144, 2304): (2, 8, 5, (2, 1)),    # 392 us
(512, 15360, 768): (3, 10, 3, (1, 3)),   # 357 us
(1024, 6144, 128): (2, 12, 1, (2, 1)),   # 170 us
(1024, 6144, 4608): (4, 8, 10, (2, 2)),  # 818 us
(512, 6144, 4608): (2, 8, 9, (1, 3)),    # 750 us
(1024, 128, 768): (6, 2, 3, (1, 3)),     # 20 us
```

### grid_88_configs (8x8 grid — AGMM + plain MM)

```python
# AGMM
(1024, 768, 4608): (20, 3, 6, (2, 2)),   # AGMM qkv — 355 us
(512, 768, 4608): (10, 3, 12, (1, 4)),   # AGMM qkv — 242 us
(1024, 768, 768): (4, 1, 16, (1, 4)),    # AGMM to_out — 190 us
(512, 768, 768): (2, 1, 16, (1, 4)),     # AGMM to_out — 132 us
# Plain MM
(1024, 6144, 4608): (4, 8, 10, (2, 2)),  # 792 us
(512, 6144, 4608): (2, 16, 6, (2, 2)),   # 718 us
(1024, 6144, 768): (4, 4, 4, (1, 4)),    # 207 us
(512, 6144, 768): (2, 8, 5, (2, 1)),     # 151 us
```

---

## End-to-End Performance

### Before Optimization (baseline)
```
Denoising (per step):  0.628s
Denoising (total):    31.42s
Total Pipeline:       32.06s
Throughput:           1.59 steps/s, 0.031 images/s
```

### After Round 1 (4 plain MM + 4 AGMM)
```
Denoising (per step):  0.614s
Denoising (total):    30.72s
Total Pipeline:       31.35s
Throughput:           1.63 steps/s, 0.032 images/s
Improvement:          ~2.2% per step
```

---

## Sweep Script Fixes for WH

1. **K_tiles for AGMM**: Divide by `tp_size` — AGMM operates on per-device K tiles (e.g., K=768, tp=8 → K_tiles=3, not 24).
2. **Trace capture disabled for AGMM on WH**: `all_gather_minimal_matmul_async` uses event synchronization internally, which is incompatible with `begin_trace_capture` on WH hardware. Plain signpost-based profiling works instead.
3. **Device config `wh_4x8`** added with: `num_links=4`, `sp_axis=0`, `tp_axis=1`, `FABRIC_1D_RING`.
