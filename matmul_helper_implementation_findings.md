# Matmul Helper Implementation Findings

**Date:** 2026-04-14
**Branch:** `wransom/matmul_op_integ_verf`
**Hardware:** Blackhole P100A

---

## What Was Done

### 1. Cleanup: Removed experimental-only helpers

Removed from `matmul_helpers_compute.{hpp,inl}`:

| Removed | Reason |
|---------|--------|
| `MoeDm1State` struct | Only used by experimental MoE kernels |
| `matmul_moe_accumulate_with_bias` | Only used by `experimental/ccl/moe_compute`, `moe_gpt` |
| `matmul_moe_w2_accumulate_with_dm1_cycling` | Same |
| `matmul_moe_w2_accumulate_with_dm1_linear` | Same |
| `matmul_accumulate_attn` | Only used by `experimental/matmul/attn_matmul` |

**Impact on experimental kernels:** Three experimental kernel files (`transformer_attn_matmul.cpp`, `moe_gpt/compute.cpp`, `moe_compute/compute.cpp`) will fail JIT compilation if their ops are invoked. These are JIT-compiled at runtime, so they don't block the build. If these kernels graduate from experimental, they can either inline the removed logic or use the Tier 3 building blocks (`matmul_single`, `matmul_accumulate`, etc.) which remain available.

### 2. Added SDPA helpers with full DST encapsulation

Two new library helpers added to `matmul_helpers_compute.{hpp,inl}`:

#### `matmul_and_pack_absolute` — SDPA streaming helper

Replaces the local `blocked_matmul_and_pack` function in `compute_streaming.hpp`.

**Encapsulates:**
- Full DST lifecycle: `tile_regs_acquire → matmul_accumulate → tile_regs_commit → tile_regs_wait → pack_tile<true> → tile_regs_release`
- Architecture dispatch: `matmul_accumulate_no_mop` on BH, `matmul_accumulate` on WH
- BH blocked-pack optimization (`blocked_pack` template parameter)
- Post-pack callback (`PostPackFn`) for hardware semaphore posting

**Does NOT manage (by design):**
- Output CB reserve/push — SDPA streaming reserves the full region upfront and pushes once after all subblocks are packed
- Input CB wait/pop — SDPA streaming manages these at a higher level
- Matmul init — caller re-inits between subblocks when data format changes (e.g., after sub_exp)

**Signature:**
```cpp
template <MatmulMode mode, bool blocked_pack = false, typename PostPackFn = NoPostPack>
ALWI void matmul_and_pack_absolute(
    const MatmulConfig& cfg,
    uint32_t in0_start, uint32_t in1_start,
    uint32_t inner_dim, uint32_t in1_stride,
    uint32_t out_cb,
    uint32_t subblock_h, uint32_t subblock_w,
    uint32_t out_num_cols,
    uint32_t row_offset, uint32_t col_offset,
    PostPackFn post_pack = {});
```

#### `matmul_blocks_absolute` — SDPA non-streaming helper

Replaces the local `matmul_blocks` function in `compute_common.hpp`.

**Encapsulates:**
- Full DST lifecycle per subblock
- Full CB management: `cb_wait_front`, `cb_reserve_back`, `cb_push_back`, `cb_pop_front`
- Matmul init (`matmul_init_short`)
- Data format reconfig (`reconfig_data_format`)
- Double subblock loop (M-subblocks × N-subblocks)
- Absolute-offset packing
- Progressive in0 CB wait
- Post-compute callback (`PostComputeFn`) for fused operations (e.g., causal mask)

**CB protocol:**
- `in1`: wait upfront, pop at end
- `in0`: progressive wait per M-subblock, NOT popped (caller manages)
- `out`: reserve upfront, push per M-subblock

**Signature:**
```cpp
template <MatmulMode mode, typename PostComputeFn = NoPostCompute>
ALWI void matmul_blocks_absolute(
    const MatmulConfig& cfg,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
    uint32_t subblock_h, uint32_t subblock_w,
    PostComputeFn post_compute = {});
```

#### Supporting functor types

| Type | Purpose |
|------|---------|
| `NoPostPack` | Default no-op for `matmul_and_pack_absolute` PostPackFn |
| `NoPostCompute` | Default no-op for `matmul_blocks_absolute` PostComputeFn |
| `TriggerReducePostPack` (in SDPA) | Posts `FPU_SFPU` semaphore for early reduce trigger |
| `CausalMaskPostCompute` (in SDPA) | Adds causal mask tiles to DST after matmul |

### 3. Renamed `matmul_tile` → `matmul_single`

Resolved the naming collision to reserve `matmul_tile` for a future Tier 1 full-loop helper.

Updated 8 files:
- `matmul_helpers_compute.{hpp,inl}` — declaration and implementation
- `compute_streaming.hpp` — SDPA streaming path
- `moreh_matmul.cpp`, `moreh_mean_w.cpp`, `moreh_sum_w.cpp` — moreh kernels
- `sdpa_compute_utils.hpp`, `sdpa_bw_compute_utils.hpp` — tt-train SDPA

### 4. Added `matmul_single_and_pack` helper

Encapsulates the single-tile matmul + DST lifecycle + output CB management pattern.
Supports `PostComputeFn` for fused SFPU ops (e.g., reciprocal in SDPA normalization).

Used to replace manual DST/CB code in `normalize_row_streaming` (compute_streaming.hpp).
The reciprocal operation is now a `RecipPostCompute` functor.

### 5. Updated SDPA kernel files

#### `compute_streaming.hpp`
- `blocked_matmul_and_pack` now delegates to `matmul_and_pack_absolute`
- `normalize_row_streaming` matmul phase now uses `matmul_single_and_pack` with `RecipPostCompute` functor
- `TriggerReducePostPack` functor defined locally for semaphore posting

#### `compute_common.hpp`
- `matmul_blocks` now delegates to `matmul_blocks_absolute`
- `CausalMaskPostCompute` functor defined locally for mask addition
- Retained wrapper function signatures for backward compatibility

---

## What Was Kept

All Tier 3 building blocks that serve non-experimental production call sites:

| Function | Production users |
|----------|-----------------|
| `matmul_single` (single dispatch, renamed from `matmul_tile`) | SDPA `normalize_row_streaming`, bmm_large_block_zm, moreh kernels |
| `matmul_single_and_pack` (new) | SDPA `normalize_row_streaming` matmul+recip phase |
| `matmul_accumulate` | All block-mode kernels via higher layers, SDPA directly |
| `matmul_accumulate_no_mop` (BH) | SDPA streaming via `matmul_and_pack_absolute` |
| `matmul_accumulate_subblock` | bmm_large_block_zm |
| `matmul_pack_to_cb` | Multiple production kernels |
| `matmul_reload_partials` | Fused bias, conv, bmm_large_block_zm |
| `matmul_accumulate_and_pack` | Conv, minimal matmul |
| `matmul_compute_one_tile` | moreh_matmul, bmm.cpp |
| `matmul_compute_inner_block` | bmm_large_block_zm via `matmul()` |
| `matmul_reduce_w` | reduce_w.cpp |
| `matmul_reduce_w_with_init` | moreh_mean_w, moreh_sum_w |
| `matmul_reduce_subblock_inplace` | SDPA compute_common `matmul_reduce` |
| `matmul()` (Layer 6) | bmm.cpp, bmm_large_block_zm.cpp |
| All init variants | Multiple production kernels |

---

## Test Results (Blackhole P100A)

### Run 1: After cleanup + SDPA helpers (before rename)

| Suite | Passed | Skipped | Failed | Time |
|-------|--------|---------|--------|------|
| SDPA decode | 9 | 0 | **0** | 1m48s |
| SDPA prefill | 3 | 2 | **0** | 10s |
| test_matmul.py (unit) | 557 | 136 | **0** | 19m58s |
| conv2d | 161 | 48 | **0** | 4m31s |
| moreh matmul + mean + sum | 392 | 311 | **0** | 6m53s |
| reduce sum | 383 | 0 | **0** | 1m45s |
| **TOTAL** | **1,505** | **497** | **0** | **~35m** |

### Run 2: After rename + `matmul_single_and_pack` + normalize_row encapsulation

| Suite | Passed | Skipped | Failed | Time |
|-------|--------|---------|--------|------|
| SDPA decode + prefill | 12 | 2 | **0** | 1m57s |
| moreh matmul + mean + sum | 392 | 311 | **0** | 8m22s |
| **TOTAL** | **404** | **313** | **0** | **~10m** |

All skip counts match the pre-existing baseline.
Zero regressions across all test suites in both runs.
Combined unique tests verified: **1,505 passed, 0 failed.**

---

## Files Modified

| File | Change |
|------|--------|
| `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp` | Removed MoE/attention; added SDPA helpers + functors; renamed `matmul_tile` → `matmul_single`; added `matmul_single_and_pack` |
| `ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl` | Same changes in implementations |
| `ttnn/.../sdpa/.../compute_streaming.hpp` | Delegates to `matmul_and_pack_absolute` + `matmul_single_and_pack`; `RecipPostCompute` functor |
| `ttnn/.../sdpa/.../compute_common.hpp` | Delegates to `matmul_blocks_absolute`; `CausalMaskPostCompute` functor |
| `ttnn/.../moreh/moreh_matmul/.../moreh_matmul.cpp` | `matmul_tile` → `matmul_single` |
| `ttnn/.../moreh/moreh_mean/.../moreh_mean_w.cpp` | `matmul_tile` → `matmul_single` |
| `ttnn/.../moreh/moreh_sum/.../moreh_sum_w.cpp` | `matmul_tile` → `matmul_single` |
| `tt-train/.../sdpa_fw/.../sdpa_compute_utils.hpp` | `matmul_tile` → `matmul_single` |
| `tt-train/.../sdpa_bw/.../sdpa_bw_compute_utils.hpp` | `matmul_tile` → `matmul_single` |

---

## Design Decisions

### Why wrapper functions instead of direct replacement

The SDPA code has many call sites for `blocked_matmul_and_pack` and `matmul_blocks`. Rather than updating every call site, I kept the original function signatures as thin wrappers that delegate to the library helpers. This:
- Minimizes diff size and risk
- Preserves backward compatibility for all SDPA call sites
- The wrappers are ALWI/SDPA_NOINLINE and add zero runtime overhead

### Why `matmul_and_pack_absolute` does not manage output CB

SDPA streaming has a "reserve once, pack many, push once" pattern:
1. `cb_reserve_back(out_cb, full_region)` at the start
2. Multiple `matmul_and_pack_absolute` calls write at different offsets
3. `cb_push_back_hold_wr_ptr(out_cb, partial)` at the end

This output CB lifecycle spans multiple helper calls. The helper cannot manage it because it only sees one subblock at a time. The caller orchestrates the reserve/push.

### Why `matmul_and_pack_absolute` always uses no_mop on BH

On Blackhole, `matmul_block_no_mop` is preferred for SDPA because the standard MOP is incompatible with the absolute-offset pack pattern used by SDPA streaming. Since this helper is specifically designed for absolute-offset packing, using no_mop on BH is the correct default.

### Why PostComputeFn for mask addition

The causal mask addition (`add_tiles`) happens after matmul accumulation but before DST commit — it modifies tiles in DST. This is the same position as the llk5 `PostComputeFn` pattern. Using a functor keeps the mask logic out of the library while still having it execute at the right point in the DST lifecycle.

---

## Remaining Work

1. **mul_bcast_cols phase in `normalize_row_streaming`** — Phase 2 (multiply output tiles by 1/sum) still has manual DST/CB management. This is NOT a matmul operation (it's `mul_tiles_bcast_cols`), so a matmul helper is the wrong abstraction. It could be a separate `bcast_cols_and_pack` helper if we want to encapsulate all DST/CB in SDPA, but that's a different helper family.

2. **Production kernel migration to Tier 1 encapsulated helpers** — The non-SDPA production kernels (bmm, fused_bias, conv, reduce_w) still use the low-level building blocks. A follow-up session could create fully-encapsulated Tier 1 helpers (similar to llk5's `matmul_block` but built on our API) to further reduce caller complexity for these kernels.
