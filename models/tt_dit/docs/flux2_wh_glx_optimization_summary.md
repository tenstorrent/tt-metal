# Flux2 WH Galaxy — Optimization Summary

**Date:** 2026-06-01
**Branch:** `flux2_wh_glxy` (forked from `friedrich/flux2`)
**Device:** WH Galaxy 4×8 (32 devices, Topology.Ring)
**Model config:** sp=4, tp=8, image 1024×1024, 50 denoising steps

---

## What We Changed

### 1. WH Galaxy device config (`wh_4x8_ring`)

Added a Wormhole Galaxy ring configuration mirroring the existing BH config:

- **File:** `models/tt_dit/utils/test.py` — added `ring_params_flux2` with `wh_4x8` mesh
- **File:** `models/tt_dit/tests/models/flux2/test_performance_flux2.py` — added `wh_glx_ring_sp0tp1_fsdp` parametrize entry
- **File:** `models/tt_dit/tests/models/flux2/test_transformer_flux2.py` — added `wh_4x8_ring` parametrize entry
- **Key settings:** `num_links=4`, `sp_axis=0`, `tp_axis=1`, `Topology.Ring`, `FSDP=True`

### 2. Matmul block size configs (`models/tt_dit/utils/matmul.py`)

All configs below were found via `sweep_mm_block_sizes.py` on WH Galaxy.

#### grid_89_configs (8×9 grid — plain MM)

| Shape (M, K, N) | M_blk | K_blk | N_blk | Subblock | Time (μs) |
|---|---|---|---|---|---|
| (1024, 6144, 2304) | 4 | 6 | 10 | (2, 2) | 451 |
| (512, 6144, 2304) | 2 | 8 | 5 | (2, 1) | 392 |
| (512, 15360, 768) | 3 | 10 | 3 | (1, 3) | 357 |
| (1024, 6144, 128) | 2 | 12 | 1 | (2, 1) | 170 |
| (1024, 6144, 4608) | 4 | 8 | 10 | (2, 2) | 818 |
| (512, 6144, 4608) | 2 | 8 | 9 | (1, 3) | 750 |
| (1024, 128, 768) | 6 | 2 | 3 | (1, 3) | 20 |

#### grid_88_configs (8×8 grid — AGMM)

| Shape (M, K, N) | Use Case | M_blk | K_blk | N_blk | Subblock | Time (μs) |
|---|---|---|---|---|---|---|
| (1024, 768, 4608) | qkv | 20 | 3 | 6 | (2, 2) | 355 |
| (512, 768, 4608) | qkv | 10 | 3 | 12 | (1, 4) | 242 |
| (1024, 768, 768) | to_out | 4 | 1 | 16 | (1, 4) | 190 |
| (512, 768, 768) | to_out | 2 | 1 | 16 | (1, 4) | 132 |

#### grid_88_configs (8×8 grid — plain MM, also used by AGMM path)

| Shape (M, K, N) | M_blk | K_blk | N_blk | Subblock | Time (μs) | Note |
|---|---|---|---|---|---|---|
| (1024, 6144, 4608) | 4 | 8 | 10 | (2, 2) | 792 | |
| (512, 6144, 4608) | 2 | 12 | 6 | (2, 2) | 721 | K_blk must divide K_tiles_per_dev=24 |
| (1024, 6144, 768) | 4 | 4 | 4 | (1, 4) | 207 | |
| (512, 6144, 768) | 2 | 8 | 5 | (2, 1) | 151 | |

> **Note:** 8×8 grid plain MM configs are shared with the AGMM code path, so `K_block` must evenly divide `K_tiles_per_device` (= K / 32 / tp_size). For K=6144, tp=8: K_tiles_per_device=24.

### 3. Sweep script WH support (`models/tt_dit/utils/sweep_mm_block_sizes.py`)

- Added `wh_4x8` device config (num_links=4, sp_axis=0, tp_axis=1)
- Added all Flux2 WH shapes (plain MM, AGMM, MM+RS)
- Fixed K_tiles calculation for AGMM (divide by tp_size for per-device K)
- Disabled trace capture for AGMM on WH (incompatible with event synchronization)
- Added MM+RS sweep code path (tensor setup + op invocation)

---

## End-to-End Performance Results

| Metric | Baseline | Round 1 | Round 2 | Round 3 (latest) | Total Improvement |
|---|---|---|---|---|---|
| **Denoising (per step)** | 0.628s | 0.614s | 0.606s | **0.557s** | **11.3%** |
| **Denoising (total, 50 steps)** | 31.42s | 30.72s | 30.31s | **27.86s** | **11.3%** |
| **Total Pipeline** | 32.06s | 31.35s | 30.93s | **28.50s** | **11.1%** |
| **Throughput** | 1.59 steps/s | 1.63 steps/s | 1.65 steps/s | **1.79 steps/s** | **12.6%** |
| Encoding | 0.161s | 0.161s | 0.161s | 0.161s | — |
| VAE Decoding | 0.448s | 0.446s | 0.446s | 0.446s | — |

### Round breakdown

- **Round 1:** 4 plain MM (8×9) + 4 AGMM (8×8) → 2.2% improvement
- **Round 2:** +4 plain MM (8×8) + 3 plain MM (8×9) → additional 1.3%
- **Round 3:** 4 MM+RS (8×7) → additional 8.1%

### sp=4/tp=8 vs sp=8/tp=4 comparison

| Metric | sp=4, tp=8 | sp=8, tp=4 |
|---|---|---|
| Denoising/step | 0.628s | 1.013s |
| Denoising total | 31.42s | 50.67s |
| Throughput | 1.59 steps/s | 0.99 steps/s |

**sp=4, tp=8 is ~38% faster** — consistent with BH results.

---

### 4. Fused MM+RS configs (`models/tt_dit/utils/matmul.py` — `fused_mmrs_configs`)

Added under `ttnn.CoreCoord(8, 9)` (WH Galaxy full grid):

| Shape (M, K, N) | M_blk | K_blk | N_blk | Subblock | Time (μs) | Default (μs) | Improvement |
|---|---|---|---|---|---|---|---|
| (1024, 2304, 6144) | 12 | 6 | 8 | (1, 4) | 953 | 1332 | 28% |
| (512, 2304, 6144) | 10 | 8 | 8 | (1, 4) | 662 | 894 | 26% |
| (1024, 3072, 6144) | 10 | 8 | 8 | (1, 4) | 1082 | — | — |
| (512, 3072, 6144) | 10 | 8 | 8 | (1, 4) | 787 | — | — |

---

## What's Still TODO

| Item | Status | Expected Impact |
| Ring SDPA tuning | Analysis done — BH has 1.53× more cores (110 vs 72 cores), chunk sizes identical | Large (likely biggest remaining gap) |
| PCC issue (88.7% on all_blocks) | Unresolved — single_blocks passes | Correctness |

---

## BH Galaxy Comparison (target)

| Metric | WH Galaxy (current) | BH Galaxy (reference) | Gap |
|---|---|---|---|
| Denoising/step | 0.557s | ~0.16s | ~3.5× |
| Denoising total | 27.86s | ~8s | ~3.5× |

The remaining gap is primarily attributed to:
1. **Ring SDPA**: BH uses 11×10 compute grid (110 cores) vs WH 8×9 (72 cores) — 1.53× core difference
2. **MM+RS**: Not yet tuned on WH
3. **Hardware**: BH has higher peak compute throughput per core
