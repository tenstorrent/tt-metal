# Phase 3, Instance 3 Results — Migration + Regression Tests

**Architecture**: Blackhole (p100a)
**Date**: 2026-03-31

---

## Summary

Migrated the production kernel `bmm_large_block_zm_fused_bias_activation.cpp` to use `matmul_block` and `add_bias_bcast_rows` helpers. All regression tests pass with zero regressions.

---

## Step 1 — Baseline (pre-migration)

Baseline from Instance 1 results (same Blackhole machine):

| Test suite | Passed | Skipped | Failed |
|-----------|--------|---------|--------|
| C++ MatmulBlockHelper (5 tests) | 5 | 0 | 0 |
| Python test_matmul.py | 556 | 136 | 0 |

Skip reasons (all pre-existing BH-specific):
- 16: transpose tile does not support tile height < 16
- 32: TinyTile Matmul needs fix on BH (Issue #31385)
- 32: Batched input not supported when bias exists
- 24: test does not support batch > 1 for sharded in/out
- 16+16: out sharded not support multiple blocks on w dim

---

## Step 2 — Production Kernel Migration

### What changed

**File**: `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

The kernel was restructured into three code paths selected at compile time:

| Condition | Code path | Matmul K-loop | Bias phase |
|-----------|-----------|---------------|------------|
| `SKIP_COMPUTE` defined | Inline K-loop | Skipped (no matmul calls) | `add_bias_bcast_rows` helper |
| `in0_transpose_tile = true` | Inline K-loop | LLK `matmul_block` + transpose interleaving | `add_bias_bcast_rows` helper |
| Otherwise (common case) | **Helper path** | `compute_kernel_lib::matmul_block` helper | `add_bias_bcast_rows` helper |

### Helper path details

For the common case (`!SKIP_COMPUTE && !in0_transpose_tile`):

**Without FUSE_BIAS**:
```cpp
compute_kernel_lib::matmul_block<
    in0_cb_id, in1_cb_id, mm_out_cb_id, mm_partials_cb_id,
    in1_transpose_tile, l1_acc, /*pack_last_to_interm=*/false, do_relu,
    PostMatmulSFPU>(...)
```
- `l1_acc`: true when `PACKER_L1_ACC` defined
- `do_relu`: true when `PACK_RELU` defined (and no bias)
- `PostMatmulSFPU`: when `SFPU_OP_INIT_ACTIVATION` defined

**With FUSE_BIAS**:
```cpp
// Phase 1: matmul packs to interm
compute_kernel_lib::matmul_block<
    in0_cb_id, in1_cb_id, out_cb_id, mm_partials_cb_id,
    in1_transpose_tile, l1_acc, /*pack_last_to_interm=*/true>(...)

// Caller: PACK_RELU config, pack format reconfig, L1_ACC disable
// Phase 2: bias add from interm to output
compute_kernel_lib::add_bias_bcast_rows<
    mm_partials_cb_id, bias_cb_id, untilize_mode_out_cb_id,
    PostBiasSFPU>(...)
```

### SFPU functor types

Added two SFPU functor structs (defined only when `SFPU_OP_INIT_ACTIVATION` is defined):
- `PostMatmulSFPU`: passed to `matmul_block` for non-bias paths
- `PostBiasSFPU`: passed to `add_bias_bcast_rows` for bias paths

Both call `SFPU_OP_FUNC_ACTIVATION` per tile, matching the production kernel's existing behavior.

### What was preserved (unchanged)

| Code section | Lines (approx) | Why kept |
|-------------|-----------------|----------|
| `transpose_tile_block` | ~40 | Single call site, in0_transpose inline path |
| `reload_from_cb_to_dst` | ~20 | Used by inline K-loop paths |
| `reblock_and_untilize` | ~30 | Untilize phase (caller-managed per design) |
| Compile-time arg parsing | ~30 | Unchanged |
| `MATMUL_DRAM_SHARDED` early exit | ~5 | Caller-managed per design |
| `get_batch_from_reader` mailbox | ~8 | Caller-managed per design |
| Untilize phase | ~20 | Caller-managed per design |
| Reconfigure-for-next-iteration | ~10 | Caller-managed per design |

### LOC comparison

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| Total file LOC | 500 | 687 | Net increase due to duplicated inline paths |
| K-loop (original, single copy) | ~170 | — | Handled all cases via nested `#ifdef` |
| K-loop (SKIP_COMPUTE inline) | — | ~170 | Preserved for SKIP_COMPUTE path |
| K-loop (transpose inline) | — | ~170 | Preserved for in0_transpose path |
| K-loop (helper path) | — | ~15 | Common case: clean helper call |
| Bias phase code | ~60 | ~20 | All paths use `add_bias_bcast_rows` helper |
| SFPU functor structs | — | ~20 | New: PostMatmulSFPU, PostBiasSFPU |

The file grew because the original had one K-loop with deeply nested `#ifdef`s handling all cases, while the migration separates into three compile-time paths. The tradeoff: more total LOC but better separation of concerns. The **common path** (no transpose, no skip_compute) — which covers the vast majority of production matmul calls — is dramatically simpler (helper call vs 170 lines of manual K-loop).

If a future `PreKBlockFn` callback is added to the helper (per Open Question 1), the in0_transpose inline path could also use the helper, eliminating ~170 lines.

---

## Step 3 — Regression Results

### Bug fix during migration

**File**: `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl:207`

The JIT compiler flagged `-Werror=sign-compare` on the expression `block < (int)num_k_blocks - 2` where `block` is `uint32_t` and the RHS is `int`. Fixed by guarding with `num_k_blocks >= 2 &&` to avoid unsigned underflow, and removing the cast:
```cpp
// Before (Instance 1):
if (block < (int)num_k_blocks - 2) {
// After (fixed):
if (num_k_blocks >= 2 && block < num_k_blocks - 2) {
```

This bug existed in Instance 1's implementation but was not caught because the existing 5 C++ tests all use `num_k_blocks=1` (single K-block, never hits the L1_ACC FIFO advancement path). The bug surfaced when the Python tests JIT-compiled the production kernel with `PACKER_L1_ACC` defined and `num_k_blocks > 1`.

### C++ Integration Tests

**Command**: `TT_METAL_HOME=$(pwd) build_Release/test/tt_metal/unit_tests_integration --gtest_filter="*MatmulBlockHelper*"`

| Test | Result |
|------|--------|
| TensixMatmulBlockHelperSingleBlock | PASSED |
| TensixMatmulBlockHelperMultiBlock | PASSED |
| TensixMatmulBlockHelperMultiSubblockBoth | PASSED |
| TensixMatmulBlockHelperSingleTileSubblock | PASSED |
| TensixMatmulBlockHelperBatch | PASSED |

**5/5 passed, 0 failed** (matches baseline).

### Python Matmul Tests

**Command**: `pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py --no-header -q`

**556 passed, 136 skipped, 0 failed** (matches baseline exactly).

### Instance 2 new tests (TensixMatmulHelper*)

Instance 2's 12 new feature tests crash with JIT segfaults on Blackhole. This is a pre-existing issue: Instance 2 developed the tests on Wormhole B0, and the JIT compilation crashes on BH with the same error pattern regardless of the migration. These failures are **not related to the migration** — they fail with the same JIT segfault before and after.

---

## Step 4 — Test Kernel Migration (Skipped)

**File**: `tests/tt_metal/tt_metal/test_kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`

**Decision**: Skipped. The test kernel uses `matmul_tiles` (tile-by-tile LLK), not `matmul_block` (sub-block LLK). The helper wraps `matmul_block`, so migration would require changing the fundamental computation approach. Additionally:
- No active integration test exercises this kernel (only perf microbenchmarks use a separate copy)
- The kernel uses 2-phase DST (`acquire_dst/release_dst`) vs helper's 4-phase
- The kernel uses a separate CB (c_25) for bias intermediate vs the production kernel's shared interm CB (c_24)
- Changing the computation approach risks breaking the perf microbenchmark comparison

---

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | **Migrated** | Production kernel: helper path for common case, inline for transpose/skip_compute |
| `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl` | **Bug fix** | Fixed sign-compare warning on line 207 (uint32 vs int comparison) |

---

## Notes

1. **Perf microbenchmark copy**: The production kernel comment says to update `tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp`. This copy should NOT be updated to use helpers because test kernels under `tests/tt_metal` can't include `kernel_lib/` headers (JIT include path limitation, per constraint 8). The copy remains as-is.

2. **Gathered variant**: `bmm_large_block_zm_fused_bias_activation_gathered.cpp` was NOT migrated in this phase. Per the design doc Section E, it's migration step 5 (after conv and test kernels). The gathered variant has `ENABLE_GLOBAL_CB` per-K-block CB pointer manipulation that the helper can't express.

3. **in0_transpose inline path**: The inline K-loop for `in0_transpose_tile=true` is a faithful copy of the original code. Per the approved decision (Open Question 1), this stays inline. If a `PreKBlockFn` callback is added to the helper in the future, this path could use the helper.

4. **SKIP_COMPUTE inline path**: The inline K-loop for `SKIP_COMPUTE` preserves the original behavior including transpose handling (just in case SKIP_COMPUTE + in0_transpose_tile are combined). This is the safest approach.
