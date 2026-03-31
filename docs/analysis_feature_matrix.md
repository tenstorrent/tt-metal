# Phase 1 Instance 1 — Feature Dimension Matrix

Comprehensive feature taxonomy of all matmul compute kernels in the codebase. 44 kernels analyzed across 7 categories.

---

## Table of Contents

1. [Production TTNN Matmul Kernels](#1-production-ttnn-matmul-kernels)
2. [Reference Test Kernels](#2-reference-test-kernels)
3. [Unit Test Matmul Kernels](#3-unit-test-matmul-kernels)
4. [Perf Microbenchmark Kernels](#4-perf-microbenchmark-kernels)
5. [Other TTNN / Experimental Matmul Kernels](#5-other-ttnn--experimental-matmul-kernels)
6. [SDPA / Attention Kernels](#6-sdpa--attention-kernels)
7. [DeepSeek V3 Unified Kernels](#7-deepseek-v3-unified-kernels)
8. [Feature Frequency Summary](#8-feature-frequency-summary)

---

## Legend

| Abbreviation | Meaning |
|---|---|
| **Strategy** | T=tile-by-tile, S=sub-blocked, L=large-block-generalized |
| **K-spill** | Uses intermediate CB for multi-block K-dimension reduction |
| **Tilize** | Fused tilize of input activations |
| **Untilize** | Fused untilize of output |
| **Bias** | R=ROW broadcast, C=COL broadcast, -=none |
| **SFPU** | Fused SFPU activation (relu/gelu/sigmoid/silu/exp/other) |
| **EltMul** | Fused eltwise multiply |
| **RELU** | PACK_RELU via llk_pack_relu_config |
| **L1ACC** | PACKER_L1_ACC (avoids spill/reload to intermediate CB) |
| **FP32** | FP32_DEST_ACC_EN (32-bit destination accumulation) |
| **Reconf** | Mixed precision / data format reconfiguration |
| **Xpose** | B matrix (in1) transpose |
| **OOO** | Out-of-order packing (pack_tile with non-sequential index) |
| **Migrated** | Y=yes (helper), P=partial, N=no |

---

## 1. Production TTNN Matmul Kernels

These are the primary migration targets — the kernels that ship in TTNN matmul ops.

| # | Kernel | LOC | Strategy | K-spill | Tilize | Untilize | Bias | SFPU | EltMul | RELU | L1ACC | FP32 | Reconf | Xpose | OOO | #ifdef | CB IDs | Migrated |
|---|--------|-----|----------|---------|--------|----------|------|------|--------|------|-------|------|--------|-------|-----|--------|--------|----------|
| 1 | `ttnn/.../matmul/.../bmm.cpp` | 61 | T | N | N | N | - | - | N | N | N | N | N | N | N | 0 | in0,in1,out | N |
| 2 | `ttnn/.../matmul/.../bmm_large_block_zm.cpp` | 28 | S | Y | N | N | - | - | N | N | N | N | N | N | N | 0 | in0,in1,out,intermed0 | Y (matmul_block) |
| 3 | `ttnn/.../matmul/.../bmm_large_block_zm_fused_bias_activation.cpp` | 500 | L | Y | N | Y | R | generic | N | Y | Y | Y | Y | Y | Y | 22 | in0,in0_transposed,in1,out,intermed0,bias | N |
| 4 | `ttnn/.../matmul/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp` | 464 | L | Y | N | Y | - | generic | N | Y | Y | Y | Y | Y | Y | 12 | in0,in1,in2,sync,sync2,mm_out_0..15,mm_partials_0..15 | N |

### Key observations — Production kernels

- **Kernel #2** (`bmm_large_block_zm.cpp`) is the only migrated kernel (28 LOC using `compute_kernel_lib::matmul_block`).
- **Kernel #3** (`bmm_large_block_zm_fused_bias_activation.cpp`) is the primary redesign target: 500 LOC, 22 `#ifdef` branches, uses nearly every feature dimension.
- **Kernel #4** (gathered variant) is similar to #3 but adds multi-device CCL gather with dynamic CB arrays (`mm_out_cb_ids[0..15]`, `mm_partials_cb_ids[0..15]`).

---

## 2. Reference Test Kernels

Test kernels under `tests/tt_metal/tt_metal/test_kernels/compute/`. These exercise various matmul patterns in isolation.

| # | Kernel | LOC | Strategy | K-spill | Tilize | Untilize | Bias | SFPU | EltMul | RELU | L1ACC | FP32 | Reconf | Xpose | OOO | #ifdef | Migrated |
|---|--------|-----|----------|---------|--------|----------|------|------|--------|------|-------|------|--------|-------|-----|--------|----------|
| 5 | `matmul.cpp` | 62 | T | N | N | N | - | - | N | N | N | N | N | N | N | 0 | N |
| 6 | `matmul_block.cpp` | 116 | S | N | N | N | - | - | N | N | N | N | Y | N | N | 4 | P |
| 7 | `matmul_large_block.cpp` | 232 | L | Y | Y | Y | - | - | N | N | N | N | N | N | N | 0 | N |
| 8 | `matmul_large_block_zm.cpp` | 103 | S | Y | N | N | - | - | N | N | N | N | N | N | N | 0 | P |
| 9 | `matmul_large_block_generalized.cpp` | 240 | L | Y | Y | Y | - | - | N | N | N | N | N | N | N | 0 | N |
| 10 | `matmul_with_bias.cpp` | 93 | T | N | N | N | R | - | N | N | N | N | N | N | N | 0 | P |
| 11 | `bmm.cpp` | 86 | T | N | N | N | - | - | N | N | N | N | N | N | N | 2 | N |
| 12 | `bmm_large_block_zm.cpp` | 105 | L | Y | N | N | - | - | N | N | N | N | N | N | N | 0 | N |
| 13 | `bmm_large_block_zm_fused_bias_activation.cpp` | 163 | L | Y | N | N | R | generic | N | N | N | N | Y | N | N | 3 | P |
| 14 | `bmm_large_block_zm_mixed_precision.cpp` | 112 | L | Y | N | N | - | - | N | N | N | N | Y | N | N | 0 | N |
| 15 | `bmm_tilize_untilize.cpp` | 272 | L | Y | Y | Y | R | generic | N | N | N | N | Y | N | N | 4 | P |
| 16 | `transformer_attn_matmul.cpp` | 86 | T | N | N | Y | - | - | N | N | N | N | N | Y | N | 0 | N |

### Key observations — Reference test kernels

- Most test kernels are simple (0 `#ifdef` branches, no fusions).
- `bmm_tilize_untilize.cpp` (#15) is the most feature-rich test kernel: tilize + untilize + bias + SFPU + mixed precision + K-spill.
- Only `matmul_block.cpp` (#6) exercises ARCH_QUASAR and WITH_DT data type reconfiguration paths.
- `transformer_attn_matmul.cpp` (#16) is the only test kernel with B-matrix transpose.

---

## 3. Unit Test Matmul Kernels

Minimal test kernels under `tests/.../unit_tests/matmul/`.

| # | Kernel | LOC | Strategy | K-spill | Tilize | Untilize | Bias | SFPU | EltMul | RELU | L1ACC | FP32 | Reconf | Xpose | OOO | #ifdef | Migrated |
|---|--------|-----|----------|---------|--------|----------|------|------|--------|------|-------|------|--------|-------|-----|--------|----------|
| 17 | `single_tile_compute.cpp` | 38 | T | N | N | N | - | - | N | N | N | N | N | N | N | 0 | N |
| 18 | `multi_tile_compute.cpp` | 57 | T | N | N | N | - | - | N | N | N | N | N | N | N | 0 | N |
| 19 | `multi_block_compute.cpp` | 76 | S | Y | N | N | - | - | N | N | N | N | N | N | N | 0 | N |

### Key observations — Unit test kernels

- These are minimal building blocks testing tile-by-tile, multi-tile, and multi-block patterns.
- Zero features beyond basic matmul + K-spill (in multi_block only).

---

## 4. Perf Microbenchmark Kernels

Perf measurement kernels under `tests/.../perf_microbenchmark/`.

| # | Kernel | LOC | Strategy | K-spill | Tilize | Untilize | Bias | SFPU | EltMul | RELU | L1ACC | FP32 | Reconf | Xpose | OOO | #ifdef | Migrated |
|---|--------|-----|----------|---------|--------|----------|------|------|--------|------|-------|------|--------|-------|-----|--------|----------|
| 20 | `1_compute_mm/.../bmm_large_block_zm_fused_bias_activation.cpp` | 169 | S | Y | N | N | R | generic | N | N | N | Y | Y | N | N | 4 | N |
| 21 | `old/matmul/.../bmm_large_block_zm_fused_bias_activation.cpp` | 162 | S | Y | N | N | R | generic | N | N | N | N | Y | N | N | 4 | N |
| 22 | `old/matmul/.../compute_local_l1.cpp` | 28 | T | N | N | N | - | - | N | N | N | N | N | N | N | 0 | N |
| 23 | `1_compute_mm/.../bmm_large_block_zm_fused_bias_activation_copy.cpp` | 380 | L | Y | N | Y | R | generic | N | Y | Y | Y | Y | N | N | 17 | N |
| 24 | `11_remote_cb_sync/.../bmm_large_block_zm_fused_bias_activation_copy.cpp` | 77 | L | Y | N | N | - | - | N | N | N | N | N | N | N | 0 | N |

### Key observations — Perf microbenchmark kernels

- Kernel #23 (`1_compute_mm/.../fused_bias_activation_copy.cpp`) is a near-clone of the production kernel (#3) with 17 `#ifdef` branches, all major features.
- Kernel #24 is a minimal version for remote CB sync testing.

---

## 5. Other TTNN / Experimental Matmul Kernels

Matmul kernels in non-standard TTNN ops and experimental operations.

| # | Kernel | LOC | Strategy | K-spill | Tilize | Untilize | Bias | SFPU | EltMul | RELU | L1ACC | FP32 | Reconf | Xpose | OOO | #ifdef | Migrated |
|---|--------|-----|----------|---------|--------|----------|------|------|--------|------|-------|------|--------|-------|-----|--------|----------|
| 25 | `conv2d/.../conv_bmm_tilize.cpp` | 640 | L | Y | Y | Y | R | generic | N | Y | Y | Y | Y | N | Y | 4 | N |
| 26 | `conv3d/.../compute.cpp` | 334 | T | Y | Y | Y | C | - | N | N | N | N | Y | N | N | 0 | P (tilize/untilize) |
| 27 | `moreh_matmul/.../moreh_matmul.cpp` | 383 | T | Y | N | N | R | - | Y | N | N | Y | Y | Y | N | 11 | N |
| 28 | `experimental/minimal_matmul/.../compute.cpp` | 422 | S | Y | N | N | R | multi | Y | Y | Y | N | Y | N | Y | 6 | P |
| 29 | `experimental/deepseek/mla/matmul_wo/.../compute.cpp` | 106 | T | N | N | N | - | - | N | N | N | N | Y | N | N | 0 | N |
| 30 | `experimental/deepseek/moe/moe_gate_mm/.../compute.cpp` | 378 | T | N | N | N | - | sigmoid | N | N | N | N | Y | N | Y | 0 | N |
| 31 | `experimental/ccl/moe_compute/.../compute.cpp` | 343 | S | Y | N | Y | - | silu | Y | N | N | N | Y | N | Y | 3 | N |
| 32 | `experimental/topk_router_gpt/.../compute.cpp` | 330 | T | Y | N | N | R | - | N | N | N | N | N | N | N | 0 | N |
| 33 | `experimental/ccl/llama_all_gather_matmul_async/.../gathered.cpp` | 418 | L | Y | N | Y | - | generic | N | Y | Y | Y | Y | Y | N | 10+ | N |
| 34 | `experimental/matmul/group_attn_matmul/.../transformer_group_attn_matmul.cpp` | 180 | T | Y | N | Y | - | - | N | N | N | N | Y | Y | N | 2 | N |
| 35 | `experimental/matmul/attn_matmul/.../transformer_attn_matmul.cpp` | 91 | T | N | N | Y | - | - | N | N | N | N | Y | Y | N | 0 | P (untilize) |

### Key observations — Other TTNN / Experimental kernels

- `conv_bmm_tilize.cpp` (#25) at 640 LOC is the most complex non-production kernel: tilize + untilize + bias + SFPU + PACK_RELU + L1_ACC + FP32 + mixed precision + out-of-order packing.
- `moreh_matmul.cpp` (#27) is unique in having both B-matrix transpose AND eltwise multiply (for masking).
- `moe_compute` (#31) chains two matmuls (W0 + W2) with SiLU + eltwise multiply in between — a multi-stage fusion pattern.
- `conv3d` (#26) is the only kernel using COL broadcast bias (all others use ROW).
- The llama_all_gather_matmul_async gathered kernel (#33) mirrors the production gathered kernel (#4) with the same feature set.

---

## 6. SDPA / Attention Kernels

Kernels where matmul is one component of a larger attention pipeline. Included because they call matmul LLK APIs, but matmul is not the sole operation.

| # | Kernel | LOC | Strategy | K-spill | Tilize | Untilize | Bias | SFPU | EltMul | RELU | L1ACC | FP32 | Reconf | Xpose | OOO | #ifdef | Migrated |
|---|--------|-----|----------|---------|--------|----------|------|------|--------|------|-------|------|--------|-------|-----|--------|----------|
| 36 | `sdpa/.../compute_streaming.hpp` | 1528 | L | Y | N | N | - | exp | Y | Y | Y | N | Y | Y | Y | 22 | P |
| 37 | `sdpa_decode/.../sdpa_flash_decode.cpp` | 662 | L | Y | Y | Y | - | exp | Y | N | N | N | Y | Y | N | 2 | P |
| 38 | `deepseek_v3/unified_kernels/flash_mla.hpp` | 747 | L | Y | N | N | - | exp | Y | N | N | N | Y | Y | N | 6 | P |

### Key observations — SDPA / Attention kernels

- These are the largest kernels (662-1528 LOC) due to online softmax, correction, and tree reduction logic.
- All use B-matrix transpose (for K^T in QK^T) and exponential SFPU (for softmax).
- `compute_streaming.hpp` (#36) is the most complex kernel in the entire codebase: 1528 LOC, 22 `#ifdef` branches, PACK_RELU, L1_ACC, out-of-order packing.
- These kernels are unlikely helper migration targets due to their multi-stage pipeline nature. The matmul portion is a small fraction of total logic.

---

## 7. DeepSeek V3 Unified Kernels

Modern unified kernel framework. These use a different architecture (BRISC/NCRISC/TRISC split in one file) and custom matmul APIs (`custom_mm_block`).

| # | Kernel | LOC | Strategy | K-spill | Tilize | Untilize | Bias | SFPU | EltMul | RELU | L1ACC | FP32 | Reconf | Xpose | OOO | #ifdef | Migrated |
|---|--------|-----|----------|---------|--------|----------|------|------|--------|------|-------|------|--------|-------|-----|--------|----------|
| 39 | `unified_kernels/matmul.hpp` | 213 | T | N | N | N | - | sigmoid/silu | N | N | N | N | Y | N | N | 3 | N* |
| 40 | `unified_kernels/dram_streaming_matmul.hpp` | 382 | S | Y | N | N | - | silu | N | N | N | Y | Y | N | N | 6 | N* |
| 41 | `unified_kernels/dram_streaming_matmul_compressed.hpp` | 314 | S | Y | N | N | - | - | N | N | N | N | Y | N | N | 2 | N* |
| 42 | `unified_kernels/dram_streaming_experts_matmul.hpp` | 393 | S | Y | N | N | - | silu | N | N | N | Y | Y | N | N | 6 | N* |
| 43 | `unified_kernels/kn_sliced_matmul.hpp` | 155 | L | N | N | N | - | - | N | N | N | N | Y | N | N | 1 | N* |
| 44 | `micro_ops/.../matmul_compressed_kernel.cpp` | 120 | L | N | N | N | - | - | N | N | N | N | Y | N | N | 2 | N* |

*N\* = These use `custom_mm_block` API (not `matmul_block` or `matmul_tiles`), a separate modern matmul path. Not candidates for kernel_lib helper migration.*

### Key observations — DeepSeek V3 unified kernels

- Use `custom_mm_block` / `custom_mm_block_init` API — a different, newer matmul path than the standard `matmul_tiles` / `matmul_block`.
- `compressed` variants use `compressed::custom_mm_compressed_block_runtime` for sparse/compressed weight format.
- Activations (sigmoid, silu) are fused on the PACK thread via semaphore coordination (`TTI_SEMWAIT`), not in the math pipeline.
- All use data format reconfiguration; `dram_streaming_matmul` and `experts_matmul` also use FP32_DEST_ACC_EN.
- These represent a modern composability approach: single-file multi-RISC kernels with compile-time args (CTArgs struct) replacing `#ifdef` trees.

---

## 8. Feature Frequency Summary

Feature frequency across all 44 kernels, ranked from most to least common.

| Rank | Feature | Count | Percentage | Kernels |
|------|---------|-------|------------|---------|
| 1 | **K-dim spill/reload** | 31/44 | 70% | All large-block and sub-blocked kernels with multi-block K |
| 2 | **Mixed precision / reconfig** | 26/44 | 59% | All production, most experimental, all DS V3, several test kernels |
| 3 | **Sub-blocked or large-block strategy** | 30/44 | 68% | Everything except simplest tile-by-tile kernels |
| 4 | **Fused SFPU activation** | 17/44 | 39% | #3,4,13,15,20,21,23,25,28,30,31,33,36,37,38,39,40,42 |
| 5 | **Fused bias (ROW broadcast)** | 12/44 | 27% | #3,10,13,15,20,21,23,25,27,28,32 + conv3d(COL) |
| 6 | **Fused untilize output** | 12/44 | 27% | #3,4,7,9,15,16,23,25,26,31,33,34,35,37 |
| 7 | **B matrix transpose** | 9/44 | 20% | #3,4,16,27,33,34,35,36,37,38 |
| 8 | **PACKER_L1_ACC** | 8/44 | 18% | #3,4,23,25,28,33,36 + DS V3 implicit |
| 9 | **PACK_RELU** | 6/44 | 14% | #3,4,23,25,33,36 |
| 10 | **FP32_DEST_ACC_EN** | 9/44 | 20% | #3,4,20,23,25,27,33,40,42 |
| 11 | **Fused tilize input** | 6/44 | 14% | #7,9,15,25,26,37 |
| 12 | **Out-of-order packing** | 8/44 | 18% | #3,4,25,28,30,31,36 |
| 13 | **Fused eltwise multiply** | 6/44 | 14% | #27,28,31,36,37,38 |
| 14 | **`#ifdef` branches (>0)** | 20/44 | 45% | Production/experimental/perf kernels |

### Feature co-occurrence in production kernel (#3)

The production kernel `bmm_large_block_zm_fused_bias_activation.cpp` uses **13 of 14** tracked features (everything except fused tilize input). This confirms it is the critical test for any helper design.

### Features by migration impact

Ranked by how many currently-unmigrated kernels each feature would help cover:

| Feature | Unmigrated kernels it would unlock | Priority |
|---------|-----------------------------------|----------|
| **PACKER_L1_ACC** | #3, #4, #25, #28, #33 (production + conv + experimental) | **Critical** |
| **Fused bias (ROW)** | #3, #25, #27, #28, #32 (production + conv + moreh + experimental) | **Critical** |
| **SFPU activation fusion** | #3, #4, #25, #28, #33 (production + conv + experimental) | **Critical** |
| **FP32_DEST_ACC_EN** | #3, #4, #25, #27, #33 (production + conv + moreh) | **High** |
| **PACK_RELU** | #3, #4, #25, #33 (production + conv) | **High** |
| **Fused untilize output** | #3, #4, #25, #33, #34 (production + conv + attn) | **High** |
| **B matrix transpose** | #3, #4, #27, #33 (production + moreh) | **Medium** |
| **Out-of-order packing** | #3, #4, #25, #28, #30, #31 (production + conv + experimental) | **Medium** |
| **Fused tilize input** | #25, #26 (conv only) | **Low** (conv-specific) |
| **COL broadcast bias** | #26 (conv3d only) | **Low** (single kernel) |
| **Fused eltwise multiply** | #27, #28, #31 (moreh + experimental) | **Low** (specialized) |

### Blocking strategy distribution

| Strategy | Count | Examples |
|----------|-------|---------|
| Tile-by-tile | 14 | matmul.cpp, bmm.cpp, matmul_with_bias.cpp, moreh_matmul.cpp, attn_matmul.cpp |
| Sub-blocked | 14 | matmul_block.cpp, bmm_large_block_zm.cpp, minimal_matmul.cpp, DS V3 streaming |
| Large-block-generalized | 16 | Production fused_bias, conv_bmm_tilize, SDPA, perf copies |

### CB ID usage patterns

Common CB assignment conventions across kernels:

| CB Index | Typical usage | Frequency |
|----------|--------------|-----------|
| c_0 / in0 | Input activation (A matrix) | 44/44 (universal) |
| c_1 / in1 | Input weights (B matrix) | 44/44 (universal) |
| c_16 / out | Output | 38/44 (most common output) |
| c_24 / intermed0 | K-dim spill/reload partials | 25/44 |
| c_3 / bias | Bias tensor | 12/44 |
| c_25 | Bias intermediate / secondary intermed | 8/44 |
| c_2 | Sync CB / secondary input / intermed | varies |
| c_26, c_27 | Untilize reblock buffers | 6/44 |

### LOC distribution

| Range | Count | Examples |
|-------|-------|---------|
| < 50 LOC | 4 | single_tile (38), multi_tile (57 - close), bmm_large_block_zm migrated (28), compute_local_l1 (28) |
| 50-100 LOC | 8 | matmul.cpp (62), bmm.cpp (86), attn_matmul (91), etc. |
| 100-200 LOC | 10 | matmul_block (116), bmm_large_block_zm (105), fused_bias test (163), etc. |
| 200-400 LOC | 12 | matmul_large_block (232), conv3d (334), moreh (383), etc. |
| 400-700 LOC | 7 | production fused_bias (500), gathered (464), conv_bmm_tilize (640), etc. |
| > 700 LOC | 3 | flash_mla (747), sdpa_flash_decode (662), compute_streaming (1528) |

**Total LOC across all 44 kernels: ~10,500 lines**

---

## 9. Full File Path Reference

For easy lookup, all 44 kernels with their full paths:

### Production TTNN (4)
1. `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp`
2. `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp`
3. `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
4. `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`

### Reference Test (12)
5. `tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp`
6. `tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp`
7. `tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block.cpp`
8. `tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp`
9. `tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_generalized.cpp`
10. `tests/tt_metal/tt_metal/test_kernels/compute/matmul_with_bias.cpp`
11. `tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp`
12. `tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm.cpp`
13. `tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`
14. `tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm_mixed_precision.cpp`
15. `tests/tt_metal/tt_metal/test_kernels/compute/bmm_tilize_untilize.cpp`
16. `tests/tt_metal/tt_metal/test_kernels/compute/transformer_attn_matmul.cpp`

### Unit Test (3)
17. `tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp`
18. `tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_tile_compute.cpp`
19. `tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp`

### Perf Microbenchmark (5)
20. `tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation.cpp`
21. `tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/bmm_large_block_zm_fused_bias_activation.cpp`
22. `tests/tt_metal/tt_metal/perf_microbenchmark/old/matmul/kernels/compute_local_l1.cpp`
23. `tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp`
24. `tests/tt_metal/tt_metal/perf_microbenchmark/11_remote_cb_sync_matmul_single_core/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp`

### Other TTNN / Experimental (11)
25. `ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp`
26. `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp`
27. `ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/moreh_matmul.cpp`
28. `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp`
29. `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp`
30. `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp`
31. `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp`
32. `ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/device/kernels/compute.cpp`
33. `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
34. `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp`
35. `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp`

### SDPA / Attention (3)
36. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp`
37. `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
38. `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`

### DeepSeek V3 Unified (6)
39. `models/demos/deepseek_v3_b1/unified_kernels/matmul.hpp`
40. `models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul.hpp`
41. `models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_matmul_compressed.hpp`
42. `models/demos/deepseek_v3_b1/unified_kernels/dram_streaming_experts_matmul.hpp`
43. `models/demos/deepseek_v3_b1/unified_kernels/kn_sliced_matmul.hpp`
44. `models/demos/deepseek_v3_b1/micro_ops/matmul_compressed/kernels/matmul_compressed_kernel.cpp`
