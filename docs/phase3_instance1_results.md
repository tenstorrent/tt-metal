# Phase 3, Instance 1 — Core Helper Implementation Results

**Architecture**: Blackhole
**Date**: 2026-03-31

---

## Summary

Implemented the two helpers from the phase 2 design doc and verified backward compatibility.

### Changes Made

#### 1. Modified `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`

- Added new template params with defaults for backward compatibility:
  - `packer_l1_acc = false` — hardware L1 accumulation
  - `pack_last_to_interm = false` — pack last K-block to interm_cb for bias pipeline
  - `pack_relu = false` — PACK_RELU on last K-block
- Removed `mm_block_init` from helper (caller responsibility per approved Q6 decision)
- All existing call sites are unchanged (new params have defaults)

#### 2. Rewritten `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl`

- Switched from 2-phase DST (`acquire_dst/release_dst`) to 4-phase DST (`tile_regs_acquire/commit/wait/release`) per design principle P3
- Replaced per-tile `pack_tile` loop with `pack_tile_block` per Open Question 4
- Replaced per-tile `copy_tile` reload with `copy_block_matmul_partials` block reload
- Added L1 accumulation code paths (when `packer_l1_acc=true`):
  - Block 0: `llk_pack_reconfig_l1_acc(0)` (no accumulation)
  - Block 1+: `llk_pack_reconfig_l1_acc(1)` (accumulate)
  - FIFO advancement between blocks: `cb_wait_front/cb_pop_front` on interm_cb
  - Reload on K-1 when `!pack_last_to_interm`, never reload when `pack_last_to_interm`
- Added `pack_last_to_interm` support: last K-block packs to `interm_cb` instead of `out_cb`
- Added `pack_relu` support: enables ZERO_RELU on last K-block when `!pack_last_to_interm`
- Pack format reconfig on last K-block when `packer_l1_acc || get_fp32_dest_acc_enabled()`

#### 3. Created `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp`

- New `add_bias_bcast_rows` helper in `compute_kernel_lib` namespace
- Template params: `partials_cb`, `bias_cb`, `out_cb`, `PostBiasFn` (default: `NoPostBias`)
- Runtime params: `in0_num_subblocks`, `in1_num_subblocks`, `out_subblock_h`, `out_subblock_w`, `bias_width_tiles`
- `PostBiasFn` fires BEFORE `tile_regs_commit()` matching production kernel SFPU placement
- Does NOT pop `bias_cb` (caller manages lifetime)
- Does NOT handle PACK_RELU, L1_ACC disable, or pack format reconfig (caller responsibility)

#### 4. Created `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.inl`

- Performs data format reconfiguration at start (`reconfig_data_format_srca/srcb`, `pack_reconfig_data_format`)
- Initializes bias broadcast: `add_bcast_rows_init_short`
- Waits for bias tiles upfront: `cb_wait_front(bias_cb, bias_width_tiles)`
- Row-broadcast bias add per subblock using `add_tiles_bcast_rows`
- Uses 4-phase DST management
- Per-tile pack (not `pack_tile_block`) to match production kernel's bias pack pattern

#### 5. Updated `bmm_large_block_zm.cpp` (production kernel)

- Added `mm_block_init(cb_in0, cb_in1, cb_intermed0, false, out_subblock_w, out_subblock_h, in0_block_w)` before the helper call

#### 6. Updated `bmm_large_block_zm.cpp` (programming examples)

- Added `mm_block_init(cb_in0, cb_in1, cb_interm, false, out_subblock_w, out_subblock_h, in0_block_w)` before the helper call

---

## Test Results

### C++ Integration Tests (matmul_block_helper)

**Command**: `build_Release/test/tt_metal/unit_tests_integration --gtest_filter="*MatmulBlockHelper*"`

| Test | Result | PCC |
|------|--------|-----|
| TensixMatmulBlockHelperSingleBlock | PASSED | 0.999458 |
| TensixMatmulBlockHelperMultiBlock | PASSED | 0.997887 |
| TensixMatmulBlockHelperMultiSubblockBoth | PASSED | 0.999304 |
| TensixMatmulBlockHelperSingleTileSubblock | PASSED | 0.999300 |
| TensixMatmulBlockHelperBatch | PASSED | (batch test) |

**5/5 passed, 0 failed** on Blackhole.

### Python Matmul Tests

**Command**: `pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -x --no-header -q`

**556 passed, 136 skipped, 0 failed** on Blackhole.

Skip reasons (all pre-existing, not related to changes):
- 16 skipped: transpose tile does not support tile height less than 16
- 32 skipped: TinyTile Matmul needs to be fixed on BH (Issue #31385)
- 32 skipped: Batched input not supported when bias exists
- 24 skipped: test does not support batch > 1 for sharded in/out
- 32 skipped: out sharded not support multiple blocks on w dim

---

## Backward Compatibility

The existing `bmm_large_block_zm.cpp` call site is unchanged:
```cpp
compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_intermed0>(
    in0_block_w, in0_num_subblocks, in1_num_subblocks, num_k_blocks,
    out_subblock_h, out_subblock_w, batch);
```

The only change to the kernel is adding `mm_block_init()` before the helper call (1 line).

All new template params have defaults matching the previous behavior:
- `packer_l1_acc = false`
- `pack_last_to_interm = false`
- `pack_relu = false`
- `PostComputeFn = NoPostCompute`

The behavioral change from 2-phase to 4-phase DST is transparent to callers and verified by the test suite.

---

## Files Changed

| File | Action |
|------|--------|
| `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` | Modified — new template params, removed mm_block_init docs |
| `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl` | Rewritten — 4-phase DST, L1_ACC, pack_tile_block, new features |
| `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp` | **New** — add_bias_bcast_rows declaration |
| `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.inl` | **New** — add_bias_bcast_rows implementation |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp` | Modified — added mm_block_init call |
| `tt_metal/programming_examples/matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp` | Modified — added mm_block_init call |

---

## Notes for Instance 2 (Tests) and Instance 3 (Migration)

- The new `packer_l1_acc`, `pack_last_to_interm`, and `pack_relu` template params are untested in isolation. Instance 2 should write tests for these feature combinations.
- The `add_bias_bcast_rows` helper is untested. Instance 2 should write composition tests (matmul_block with pack_last_to_interm → add_bias_bcast_rows).
- Instance 3 can use the helpers to migrate `bmm_large_block_zm_fused_bias_activation.cpp`. The caller-managed items (PACK_RELU config, L1_ACC disable, format reconfig between phases) match the production kernel's existing pattern.
- The bias helper uses per-tile `pack_tile` (not `pack_tile_block`) because the production kernel's bias phase uses per-tile packing. If pack_tile_block is preferred for consistency, it's a 1-line change.
