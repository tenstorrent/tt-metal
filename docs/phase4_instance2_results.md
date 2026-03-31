# Phase 4, Instance 2 Results — PreKBlockFn Callback + Migrations

**Architecture**: Blackhole (p100a)
**Date**: 2026-03-31

---

## Summary

Added `PreKBlockFn` callback to `matmul_block`, migrated the production kernel's in0_transpose path to use the helper (eliminating ~155 lines of inline K-loop code), and verified all BH test failures from Phase 3 are resolved.

---

## Part A — PreKBlockFn Added to matmul_block

### Files changed

| File | Change |
|------|--------|
| `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` | Added `NoPreKBlock` struct, `PreKBlockFn` template param, `pre_k_block` runtime param |
| `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl` | Added `pre_k_block(block, num_k_blocks, last_out)` call between PACK_RELU and cb_wait_front |

### API change

```cpp
template <
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,
    bool pack_relu = false,
    typename PostComputeFn = matmul_block_config::NoPostCompute,
    typename PreKBlockFn = matmul_block_config::NoPreKBlock>  // NEW
ALWI void matmul_block(
    ...,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {});  // NEW
```

The new template parameter and runtime parameter are both last, with defaults, so **all existing call sites are unchanged**.

### Callback contract

- Called once per K-block, BEFORE `cb_wait_front` for inputs, AFTER PACK_RELU config
- Signature: `void operator()(uint32_t block, uint32_t num_k_blocks, bool is_last)`
- Default `NoPreKBlock` is a no-op (compiled away)

---

## Part B — in0_transpose Path Migrated

### Files changed

| File | Change |
|------|--------|
| `bmm_large_block_zm_fused_bias_activation.cpp` | Added `TransposePreKBlock` struct, replaced ~155-line inline K-loop with helper call |

### How it works

`TransposePreKBlock` is a zero-state functor defined at file scope with 9 template parameters (all constexpr CB IDs and dimensions). Per K-block, it:

1. `transpose_wh_init_short(in0_transpose_cb_id)` — init WH transpose
2. `PACK((pack_reconfig_data_format(in0_cb_id)))` — reconfig pack for transposed CB
3. `PACK((llk_pack_reconfig_l1_acc(0)))` — disable L1_ACC during transpose packing (only when `PACKER_L1_ACC`)
4. `transpose_tile_block<in0_block_num_tiles>(in0_transpose_cb_id, in0_cb_id)` — transpose input block
5. `mm_block_init_short(...)` — reinit matmul for transposed data
6. `PACK((pack_reconfig_data_format(mm_partials_cb_id)))` — restore pack format

After the functor returns, the helper's standard K-loop continues: `cb_wait_front` (for the now-transposed tiles), subblock matmul, pack with L1_ACC management, etc.

### L1_ACC compatibility

The helper's non-last-block L1_ACC management handles the transpose case correctly:
- Block 0: `llk_pack_reconfig_l1_acc(0)` (no accumulation)
- Block 1+: `llk_pack_reconfig_l1_acc(1)` (accumulate)

The transpose functor disables L1_ACC for the transpose pack phase, and the helper re-enables it for the matmul pack phase. This matches the original inline code's `else if (in0_transpose_tile) { llk_pack_reconfig_l1_acc(1) }` behavior.

### Code path unification

The production kernel now uses `std::conditional_t` to select the PreKBlockFn:

```cpp
using PreFn = std::conditional_t<in0_transpose_tile, XposeFn, NoPreFn>;
```

Both transpose and non-transpose paths call the same helper with different PreKBlockFn types. This eliminates the code duplication.

### LOC comparison

| Metric | Before (Phase 3) | After (Phase 4) | Delta |
|--------|-------------------|------------------|-------|
| Total file LOC | 687 | ~540 | -147 |
| SKIP_COMPUTE inline K-loop | ~170 | ~170 | 0 (unchanged) |
| in0_transpose inline K-loop | ~155 | 0 | -155 (eliminated) |
| TransposePreKBlock struct | 0 | ~20 | +20 |
| Helper call (common path) | ~15 | ~20 | +5 (PreFn added) |

Net reduction: **~130 lines** from the production kernel.

### What's left inline

| Code section | Lines | Why kept inline |
|--------------|-------|-----------------|
| `SKIP_COMPUTE` K-loop | ~170 | Degenerate case (no matmul), can't use helper |
| `transpose_tile_block` function | ~40 | Used by both SKIP_COMPUTE and TransposePreKBlock |
| `reload_from_cb_to_dst` function | ~20 | Used by SKIP_COMPUTE inline path |
| `reblock_and_untilize` function | ~30 | Untilize phase (caller-managed) |

---

## Part C — conv_bmm_tilize.cpp Assessment

**Decision**: NOT migrated. The conv kernel has fundamental incompatibilities with both helpers:

### Matmul K-loop incompatibilities

1. **Direct FIFO pointer management**: The conv kernel resets partials CB FIFO read/write pointers (`get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr`) after each K-block. This bypasses the standard CB API that the helper relies on. The kernel uses "in-place" accumulation where partials are read and written at the same FIFO position — the helper's standard `cb_push_back/cb_pop_front` would advance the pointers and break this pattern.

2. **Interleaved tilize/matmul phases**: Per K-block, the kernel optionally tilizes input (`tilize_in<...>`) before the matmul computation. This includes `mm_block_init_short_with_both_dt` (a 4-argument variant that the helper doesn't call) and `pack_reconfig_data_format` / `pack_reconfig_l1_acc` transitions between tilize and matmul phases.

3. **Dynamic output CB switching**: `curr_matmul_out_cb` changes between `matmul_partials_cb` and `mm_out_cb_id` at runtime based on whether it's the last K-block. The helper selects the pack target at compile time via `pack_last_to_interm`.

4. **Different L1_ACC management**: The conv kernel uses `pack_reconfig_l1_acc(fuse_bias ? 1 : 0)` on the last block (accumulates for bias, stops for output), which differs from the helper's two-branch pattern.

### Bias phase incompatibilities

1. **Bias block offset**: The conv kernel iterates over `in1_num_blocks_w` blocks along the W dimension, with `bias_block_offset += in1_block_w` per block. The bias helper always starts from bias index 0.

2. **Upfront partials wait**: The conv kernel waits for ALL output tiles at once (`cb_wait_front(matmul_partials_cb, out_block_num_tiles)`), while the helper waits per subblock.

3. **Different reconfig pattern**: Uses `reconfig_data_format(4 args)` vs the helper's separate `reconfig_data_format_srca` / `reconfig_data_format_srcb` calls.

---

## Part D — Gathered Variants Assessment

**Decision**: NOT migrated. Both gathered variants have incompatibilities that prevent direct helper usage:

### Original gathered variant (`matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`)

1. **Variable inner dimension**: `unpadded_in0_block_w = unpadded_in0_shard_widths_in_tiles[curr_ring_idx]` changes per K-block based on ring position. The helper uses a single `block_w` for all K-blocks.

2. **Different input CB per block**: `input0_cb_id = block == 0 ? in0_cb_id : in2_cb_id`. The helper's `in0_cb` is a compile-time template parameter.

3. **Different spill condition**: `spill = num_blocks > 1 && (out_block_num_tiles / out_subblock_num_tiles) > 1`. The helper spills whenever `num_k_blocks > 1`.

4. **`ENABLE_GLOBAL_CB` operates after cb_wait_front**: The CB pointer manipulation (`calculate_next_block_index_and_update_rd_ptr`) runs AFTER `cb_wait_front`, while `PreKBlockFn` fires BEFORE.

5. **Per-batch CB arrays**: Uses `mm_out_cb_ids[b]` and `mm_partials_cb_ids[b]` (different CBs per batch). The helper uses compile-time CB IDs.

### Experimental CCL gathered variant

Same as the original gathered variant except:
- Uses a single input CB (`input0_cb_id = in0_cb_id`)
- Uses fixed inner dim (`unpadded_in0_block_w = in0_block_w`)

But still has: different spill condition, `ENABLE_GLOBAL_CB` after waits, per-batch CB arrays.

---

## Part E — BH Test Failures RESOLVED

All **12 TensixMatmulHelper*** tests now pass on Blackhole (p100a). The JIT compilation segfaults reported in Phase 3 Instance 3 no longer reproduce on the current branch state. No code changes were needed — the fix was already present (likely the sign-compare bug fix from Phase 3 Instance 3 on `matmul_block_helpers.inl:207`).

### All 12 feature tests on BH

| Test | PCC | Result |
|------|-----|--------|
| TensixMatmulHelperL1AccSingleBlock | 0.999287 | PASSED |
| TensixMatmulHelperL1AccMultiBlock | 0.997920 | PASSED |
| TensixMatmulHelperPackRelu | 0.999302 | PASSED |
| TensixMatmulHelperPackReluMultiBlock | 0.999292 | PASSED |
| TensixMatmulHelperFusedBias | 0.998741 | PASSED |
| TensixMatmulHelperFusedBiasMultiBlock | 0.998700 | PASSED |
| TensixMatmulHelperL1AccBias | 0.998507 | PASSED |
| TensixMatmulHelperBiasRelu | 0.998703 | PASSED |
| TensixMatmulHelperL1AccBiasRelu | 0.998479 | PASSED |
| TensixMatmulHelperMultiSubblockBias | 0.999134 | PASSED |
| TensixMatmulHelperMultiSubblockL1AccBias | 0.998548 | PASSED |
| TensixMatmulHelperL1AccRelu | 0.997887 | PASSED |

---

## Part F — Regression Test Results

### C++ Integration Tests

**Command**: `TT_METAL_HOME=$(pwd) build_Release/test/tt_metal/unit_tests_integration --gtest_filter="*MatmulBlockHelper*:*TensixMatmulHelper*"`

| Test suite | Passed | Failed |
|-----------|--------|--------|
| TensixMatmulBlockHelper* (5 existing) | 5 | 0 |
| TensixMatmulHelper* (12 feature tests) | 12 | 0 |
| **Total** | **17** | **0** |

### Python Matmul Tests

**Command**: `pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py --no-header -q`

**556 passed, 136 skipped, 0 failed** (matches Phase 3 baseline exactly).

Skip reasons (all pre-existing BH-specific):
- 16: transpose tile does not support tile height < 16
- 32: TinyTile Matmul needs fix on BH (Issue #31385)
- 32: Batched input not supported when bias exists
- 24: test does not support batch > 1 for sharded in/out
- 16+16: out sharded not support multiple blocks on w dim

---

## Files Changed (Summary)

| File | Action | Description |
|------|--------|-------------|
| `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` | **Modified** | Added `NoPreKBlock`, `PreKBlockFn` template param, `pre_k_block` runtime param |
| `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl` | **Modified** | Added `pre_k_block()` call in K-loop |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | **Modified** | Added `TransposePreKBlock`, replaced inline in0_transpose K-loop with helper call |
