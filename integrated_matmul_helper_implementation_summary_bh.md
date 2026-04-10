# Integrated Matmul Helper Implementation Summary (Blackhole Verified)

**Branch base:** `wransom/matmulop_bh_impl` | **Verified on:** Blackhole P100A | **Date:** 2026-04-10

---

## Overview

This document describes the merge analysis and verification of the matmul helper library
implementations from two independent Claude Code instances — one running on Wormhole
(`wransom/matmulop_wh_impl`) and one on Blackhole (`wransom/matmulop_bh_impl`). Both
migrated the `MatmulOp<IsBlockMode>` class from `matmul_op.h` into a free-function helper
library at `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.{hpp,inl}`.

---

## Merge Analysis

### Branch Structure

Both branches share the **same merge-base** (`3a2a210ec2`) and all intermediate commits.
The only differences are the final tip commits:

- **WH tip:** `7341e86eab matmul op compute helper implementation for wormhole`
- **BH tip:** `9c8820b19b matmul op compute helper implementation for blackhole`

Both modify the **same 45 files** (excluding summary `.md` files).

### Differences Between Branches

All differences are **naming/style only** — no semantic differences in any code path:

| Aspect | WH Branch | BH Branch (used) |
|--------|-----------|-------------------|
| Config construction | Member-by-member init (8-10 lines) | `MatmulConfig::block(...)` / `::tile(...)` factories |
| Template shorthands | `MatmulMode::TILE` / `MatmulMode::BLOCK` | `TILE` / `BLOCK` (convenience aliases) |
| Namespace usage | `compute_kernel_lib::` prefix everywhere | `using namespace compute_kernel_lib;` at file scope |
| DST acquire | `matmul_acquire_dst()` wrapper | `tile_regs_acquire()` directly |
| `init_short_with_both_dt` | Non-templated | `template<MatmulMode mode>` with `static_assert` |
| `accumulate_no_mop` | Non-templated | `template<MatmulMode mode>` with `static_assert` |
| Reduce-W variants | `matmul_reduce_w<bool reinit_per_tile>` | Separate `matmul_reduce_w` / `matmul_reduce_w_with_init` |
| Pack functions | `matmul_pack_output` / `matmul_pack_partials` | `matmul_pack_to_cb` / `matmul_pack_to_partials` |
| Compute tile | `matmul_compute_tile` | `matmul_compute_one_tile` |
| Inner block | `matmul_inner_block` | `matmul_compute_inner_block` |
| Attention | `matmul_attention` | `matmul_accumulate_attn` |
| MoE functions | `matmul_moe_with_bias` | `matmul_moe_accumulate_with_bias` |
| ALWI handling | `#include "api/compute/matmul.h"` in `.hpp` | `#ifndef ALWI` / `#define ALWI` guard in `.hpp` |

### Why the BH Branch Is the Correct Merge

1. **All WH bug fixes are present** — the shared commit `ddc136db09 wormhole bug fixes`
   (which fixed the SDPA `matmul_reduce_subblock_inplace` deadlock) is in both branches'
   history.

2. **Correct `#ifdef ARCH_BLACKHOLE` / `#else` guards** — all architecture-specific code
   paths are properly guarded. The `#else` (WH) paths are identical between both branches.

3. **Properly templated `matmul_reduce_subblock_inplace`** — the critical SDPA fix from
   the WH instance (making this function templated on `MatmulMode` instead of hardcoding
   `TILE`) is present in the BH branch.

4. **Cleaner API** — factory methods, convenience aliases, and `static_assert` on
   block-mode-only functions are more robust than the WH variants.

---

## Files Created

| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp` | Declarations, config structs, enums, documentation |
| `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl` | All function implementations (included at bottom of .hpp) |

## Files Migrated (~28 production kernels)

**Core matmul:**
- `bmm.cpp`, `bmm_large_block_zm.cpp`, `bmm_large_block_zm_fused_bias_activation.cpp`,
  `bmm_large_block_zm_fused_bias_activation_gathered.cpp`

**SDPA:**
- `compute_streaming.hpp`, `compute_common.hpp`

**Attention:**
- `transformer_attn_matmul.cpp`, `transformer_group_attn_matmul.cpp`

**Reduction/Moreh:**
- `reduce_w.cpp`, `moreh_matmul.cpp`, `moreh_mean_w.cpp`, `moreh_sum_w.cpp`

**Conv:**
- `conv_bmm_tilize.cpp`, `conv3d/compute.cpp`

**Deepseek/MoE:**
- `matmul_wo/compute.cpp`, `moe_gate_mm/compute.cpp`, `moe_compute/compute.cpp`,
  `moe_gpt/compute.cpp`

**CCL/Minimal:**
- `minimal_matmul/compute.cpp`, `all_gather_minimal_matmul_async/compute.cpp`,
  `llama_all_gather_matmul_async/.../bmm_large_block_zm_fused_bias_activation_gathered.cpp`

**Other:**
- `topk_router_gpt/compute.cpp`, `bmm_tilize_untilize.cpp`

**tt-train:**
- `sdpa_fw_compute_kernel.cpp`, `sdpa_compute_utils.hpp`, `sdpa_bw_compute_utils.hpp`

**Models:**
- `deepseek_v3_b1/unified_kernels/rope.hpp`

## Files Removed (cleanup)

| File | Reason |
|------|--------|
| `.matmul_op_project/` (entire directory) | PoC migration examples and design docs — no longer needed |
| `tt_metal/hw/inc/api/compute/matmul_op.h` | New file added by this branch, no kernel includes it |

---

## API Design

**Namespace:** `compute_kernel_lib`

**Key types:**
- `MatmulMode` enum: `TILE` or `BLOCK` — template parameter for compile-time LLK dispatch
- `MatmulConfig` struct: replaces `ckernel::MatmulOpConfig`, with `::tile()` and `::block()` factory methods
- `MatmulBlockShape` struct: dimensions for the fully automated `matmul()` function
- `MoeDm1State` struct: state for MoE W2 accumulate helpers

**6-layer function hierarchy:**

| Layer | Key Functions | Replaces |
|-------|--------------|----------|
| 0 - Init | `matmul_init`, `matmul_init_short`, `*_with_dt`, `*_with_both_dt` | `mm.init()`, `mm.init_short()`, etc. |
| 1 - Single op | `matmul_tile` | `mm.matmul_one_tile()` |
| 2 - Accumulate | `matmul_accumulate`, `*_subblock`, `*_no_mop` | `mm.accumulate()`, `mm.accumulate_tile_subblock()` |
| 3 - Pack/reload | `matmul_pack_to_cb`, `*_to_partials`, `matmul_reload_partials` | `mm.end_to_output()`, `mm.end_to_partials()` |
| 4 - Compound | `matmul_accumulate_and_pack`, `matmul_compute_one_tile`, `matmul_compute_inner_block` | `mm.accumulate_and_pack()`, `mm.compute_one_tile()` |
| 5 - Specialized | `matmul_reduce_w`, `*_with_init`, `*_attn`, `*_reduce_subblock_inplace`, `matmul_moe_*` | All specialized MatmulOp methods |
| 6 - Automated | `matmul(cfg, shape)` | `mm.run()` |

---

## Architecture-Specific Considerations

1. **`matmul_accumulate_no_mop`** — Guarded by `#ifdef ARCH_BLACKHOLE`. Calls
   `ckernel::matmul_block_no_mop()` which only exists on Blackhole. The SDPA streaming
   kernel uses this in `#ifdef ARCH_BLACKHOLE` / `#else` branches.

2. **`mm_no_mop_init_short` / `mm_no_mop_reinit_short`** — Direct LLK calls in
   `compute_streaming.hpp` (not through the helper library), guarded by
   `#ifdef ARCH_BLACKHOLE`. The `#else` paths use `mm_block_init_short`.

3. **CB count limits** — 32 on Wormhole, 64 on Blackhole. No helper-level concern
   (CB IDs are passed through).

4. **`matmul_init_short_with_both_dt`** — Block mode only, enforced via
   `static_assert(mode == MatmulMode::BLOCK)`.

---

## Bugs Found and Fixed (by WH and BH instances)

### 1. SDPA Deadlock — `matmul_reduce_subblock_inplace` (found on WH)

**Problem:** The initial implementation hardcoded `MatmulMode::TILE` in
`matmul_reduce_subblock_inplace`. The original `MatmulOp::reduce_subblock_inplace` was a
member of the templated class, inheriting whatever mode the class was instantiated with.
SDPA uses it with block-mode configuration, but the free function called `matmul_tiles()`
instead of `matmul_block()`, causing all 64 cores to hang.

**Diagnosis:** Watcher logs showed all worker cores with BRISC in `CWFW`, NCRISC in `CRBW`,
and TRISC2 in `K` — a compute pipeline deadlock. Bisecting isolated the bug to
`compute_common.hpp`'s `matmul_reduce` calling `matmul_reduce_subblock_inplace`.

**Fix:** Made `matmul_reduce_subblock_inplace` templated on `MatmulMode`. Call sites pass
`<MatmulMode::BLOCK>`. Fixed in shared commit `ddc136db09`.

### 2. ALWI Macro Ordering (found on BH)

**Problem:** The `.hpp` forward declarations used `ALWI` before it was defined by
`common_globals.h` (which only comes in through the `.inl`). When
`matmul_helpers_compute.hpp` is the first include in a kernel file, compilation failed.

**Fix (BH):** Added `#ifndef ALWI` / `#define ALWI inline __attribute__((always_inline))`
at the top of the `.hpp`.

**Fix (WH):** Added `#include "api/compute/matmul.h"` in the `.hpp` to bring in the real
`ALWI` definition.

**Merged approach:** Uses the BH `#ifndef ALWI` guard (more self-contained, no extra includes
in the declarations header).

---

## Test Results — Blackhole P100A (Integration Verification)

These tests were run after the merge analysis on Blackhole with a cleared JIT cache.

| Suite | Passed | Skipped | Failed | Time |
|-------|--------|---------|--------|------|
| test_matmul.py (unit) | 557 | 136 | **0** | 11m47s |
| SDPA decode + prefill | 12 | 2 | **0** | 1m49s |
| moreh matmul + mean + sum + reduction sum | 775 | 311 | **0** | 6m40s |
| conv2d | 161 | 48 | **0** | 3m54s |
| **Total** | **1,505** | **497** | **0** | **24m10s** |

All skips are pre-existing (BH-specific limitations, config exclusions, grid size
requirements). No test was skipped due to our changes.

### Migrated Kernels Exercised During Testing

The following migrated compute kernels were observed compiling and executing via the Watcher:

- `bmm_large_block_zm_fused_bias_activation.cpp` — core matmul tests
- `sdpa_flash_decode.cpp` → includes `compute_streaming.hpp` — SDPA decode tests
- `sdpa.cpp` → includes `compute_common.hpp` + `compute_streaming.hpp` — SDPA prefill tests
- `moreh_matmul.cpp` — moreh matmul tests
- `moreh_mean_w.cpp` — moreh mean tests (exercises `matmul_reduce_w_with_init`)
- `moreh_sum_w.cpp` — moreh sum tests (exercises `matmul_reduce_w_with_init`)
- `reduce_w.cpp` — reduction sum tests (exercises `matmul_reduce_w`)
- `conv_bmm_tilize.cpp` — conv2d tests (exercises `matmul_init_short_with_both_dt<BLOCK>`)

---

## Previous Test Results (from individual instances)

### Wormhole n150 (WH instance — 0 failures)

| Suite | Passed | Time |
|---|---|---|
| test_matmul.py | 589 | 11m46s |
| test_sum.py | 383 | 3m04s |
| test_moreh_matmul.py | 89 | 4m36s |
| test_moreh_mean.py | 76 | 5m12s |
| test_moreh_sum.py | 227 | 1m44s |
| test_reduction_mean.py | 175 | 59s |
| test_sdpa_decode.py | 9 | ~5s each |
| test_sdpa_prefill.py | 3 | 8.91s |
| **Total** | **1,551** | |

### Blackhole P100A (BH instance — 0 failures)

| Suite | Passed | Skipped | Failed |
|-------|--------|---------|--------|
| test_matmul.py (main unit) | 557 | 136 | **0** |
| test_linear + addmm + experimental + batch_mismatch | 482 | 35 | **0** |
| test_sparse_matmul | 11 | 0 | **0** |
| test_conv2d | 161 | 48 | **0** |
| nightly matmul (11 test files) | 1,147 | 994 | **0** |
| moreh_matmul (nightly) | 89 | 84 | **0** |
| SDPA prefill | 3 | 2 | **0** |
| SDPA decode | 9 | 0 | **0** |
| moreh_mean + moreh_sum | 303 | 227 | **0** |
| **Total** | **2,762** | **1,526** | **0** |

---

## What Remains

1. **Wormhole validation** — Run the integrated branch on Wormhole hardware to verify
   the BH branch's code works correctly on WH. The `#else` (non-BLACKHOLE) code paths
   were tested by the WH instance but against a different (naming-only) variant of the
   helper library.

2. **MoE/CCL kernels** — The MoE kernels (`moe_compute`, `moe_gpt`, `topk_router_gpt`)
   were migrated but require multi-device setups (T3000/Galaxy) for device-level validation.
   Compilation succeeded and the helper calls are structurally identical.

3. **tt-train kernels** — `sdpa_fw_compute_kernel.cpp`, `sdpa_compute_utils.hpp`, and
   `sdpa_bw_compute_utils.hpp` were migrated but tt-train tests were not run.
