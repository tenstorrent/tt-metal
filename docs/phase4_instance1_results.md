# Phase 4, Instance 1 Results — matmul_tile Helper + Migrations

## Architecture

Tests run on **Blackhole** (p100a board, hostname bh-34-special).

## Summary

Implemented a minimal `matmul_tile` helper for the tile-by-tile matmul pattern (~48 call sites in the codebase) and migrated the production `bmm.cpp` kernel to use it. All existing tests pass with no regressions.

## Deliverables

### 1. New helper: `matmul_tile_helpers.hpp/inl`

**Files created:**
- `ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp`
- `ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.inl`

**API signature:**
```cpp
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    bool transpose = false,
    typename PostComputeFn = matmul_tile_config::NoPostCompute>
ALWI void matmul_tile(
    uint32_t Mt, uint32_t Nt, uint32_t Kt,
    uint32_t batch = 1,
    PostComputeFn post_compute = {});
```

**Design decisions (per orchestration doc constraints):**
- **No enums** — the prior attempt's WaitMode, InitUninitMode, ReconfigureRegisterDatatypeMode were all removed per PR feedback. This version has zero enums.
- **No param structs** — flat params only (constraint 9).
- **Caller calls `mm_init`** — consistent with `matmul_block` where caller calls `mm_block_init`.
- **4-phase DST management** — `tile_regs_acquire/commit/wait/release`, matching `matmul_block` and all other kernel_lib helpers.
- **PostComputeFn** — fires per output tile after Kt accumulation, before packing. Same pattern as `matmul_block`.
- **CB sync: wait-per-tile** — `cb_wait_front(1)` + `cb_pop_front(1)` per Kt iteration for both inputs. Matches the production `bmm.cpp` pattern.
- **Loop order: batch × Mt × Nt × Kt** — must match CB production order from reader.

### 2. Migration: `bmm.cpp`

**File modified:** `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp`

Replaced 26 lines of manual matmul loop code with a single helper call (7 lines total including includes). The kernel now:
1. Reads compile-time args (batch, Mt, Kt, Nt) and named CB indices
2. Calls `mm_init(cb_in0, cb_in1, cb_out)` (caller-managed)
3. Calls `compute_kernel_lib::matmul_tile<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt, batch)`

Removed dependency on `experimental/circular_buffer.h` and `api/compute/tile_move_copy.h` (the helper includes them internally).

### 3. moreh_matmul.cpp assessment

The simple `matmul()` path (lines 316-330) is a direct match for the helper pattern. However, migration was deferred because:
- The function is only ~15 lines — minimal code savings
- It includes `FP32_DEST_ACC_EN` pack_reconfig_data_format handling that the helper doesn't provide
- Adding FP32_DEST_ACC_EN handling to the helper would violate constraint 10 ("only add configurability with real call sites") since no other tile-by-tile call site needs it
- The complex `matmul_with_transpose_and_mask()` path cannot use the helper (mask/transpose interleaving)

**Recommendation:** Revisit moreh_matmul migration if FP32_DEST_ACC_EN support is added to the helper in a future phase (when more call sites need it).

### 4. Isolated tests

**Files created:**
- `tests/tt_metal/tt_metal/test_kernels/compute/test_matmul_tile_helper_compute.cpp` — compute kernel using the helper
- `tests/tt_metal/tt_metal/test_kernels/dataflow/writer_matmul_tile_sequential.cpp` — simple sequential DRAM writer
- `tests/tt_metal/tt_metal/integration/matmul/test_matmul_tile_helper.cpp` — host test with 6 test cases
- `tests/tt_metal/tt_metal/integration/sources.cmake` — updated to include new test

**Test cases:**

| Test | Config | PCC |
|------|--------|-----|
| Basic1x1x1 | M=32, N=32, K=32, batch=1 | 0.999742 |
| KAccum | M=32, N=32, K=128, batch=1 | 0.998106 |
| MultiTile | M=64, N=64, K=64, batch=1 | 0.999289 |
| NonSquare | M=32, N=64, K=96, batch=1 | 0.998980 |
| Batch | M=32, N=32, K=64, batch=2 | 0.999316 |
| LargerBatch | M=64, N=32, K=64, batch=3 | 0.999291 |

All 6 tests pass with PCC > 0.97 threshold (actual PCC > 0.998 in all cases).

## Regression Test Results

### C++ integration tests (unit_tests_integration)

**matmul_tile helper tests:** 6/6 passed
**matmul_block helper tests:** 5/5 passed
**matmul_helper_features tests:** (run by instance 2)
**All matmul integration tests:** 22 passed, 1 skipped (pre-existing skip: `TensixMatmulSingleCore` — "Fast dispatch buffer memory issue")

### Python matmul tests

```
tests/ttnn/unit_tests/operations/matmul/test_matmul.py
556 passed, 136 skipped, 0 failures (812s)
```

All 136 skips are pre-existing (BH-specific: TinyTile issues, batch+bias, sharded limitations).

**No regressions from the bmm.cpp migration.**

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp` | Created | Helper header with types + declaration |
| `ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.inl` | Created | Helper implementation |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp` | Modified | Migrated to use matmul_tile helper |
| `tests/.../compute/test_matmul_tile_helper_compute.cpp` | Created | Test compute kernel |
| `tests/.../dataflow/writer_matmul_tile_sequential.cpp` | Created | Test writer kernel |
| `tests/.../integration/matmul/test_matmul_tile_helper.cpp` | Created | Host test file (6 tests) |
| `tests/.../integration/sources.cmake` | Modified | Added new test to build |
