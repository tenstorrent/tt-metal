# Bug Report: [TT-Train] LayerNorm: Incorrect Accumulation for Large Feature Dimensions

## Summary

The TTML LayerNorm backward kernel (`layernorm_bw_kernel.cpp`) has a **critical accumulation bug** in `compute_dy_gamma_sum()` that causes ~99% of data to be lost when processing tensors with large feature dimensions that don't fit in L1 cache.

**Status:** The original NIGHTLY test passes on fresh main due to random mean-zero data masking the bug. Deterministic tests definitively expose the bug.

## Bug Description

### Root Cause

In `compute_dy_gamma_sum()` (block-based path for large tensors):

```cpp
// BUGGY: Each iteration OVERWRITES sum_register instead of accumulating
for (uint32_t col = 0; col < Wt; ++col) {
    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, sum_register);  // OVERWRITES!
}
// Only the last tile uses add_binary_tile for accumulation
add_binary_tile(sum_register, working_register, sum_register);  // Line ~280
```

**Result:** For `Wt=265` tiles (8462 features), only the last 2 tiles contribute to the sum. ~99% of data is lost.

### Correct Pattern (from TTNN)

```cpp
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    if (wt > 0) {
        copy_tile(cb_accumulate, 0, dst0);  // Load previous accumulator
    }
    reduce_tile(..., dst0);  // Accumulates into dst0
    pack_tile(dst0, cb_accumulate);  // Save for next iteration
}
```

### Why Random Tests Pass

The original NIGHTLY test uses random uniform[-1,1] data which has mean≈0. When most tiles are lost:
- Correct sum: expected value ≈ 0 (mean-zero data)
- Buggy sum: expected value ≈ 0 (still mean-zero, just fewer samples)
- With loose tolerance (atol=0.5), the error often stays within bounds

**Deterministic tests with constant positive values expose the bug definitively:**
- Constant inputs: dy=1.0, gamma=1.0, x=0.5
- Correct sum: ~8192 (for 8192 features)
- Buggy sum: ~64 (only last 2 tiles)
- Error: max_diff ≈ 1000

## Reproduction Steps

### Prerequisites

- Wormhole n150 device
- tt-metal with tt-train

### Native Build Reproduction (Verified)

**This is the verified reproduction method.** Fresh clone and standalone tt-train build on physical machine.

#### Step 1: Clone and Build tt-metal

```bash
# 1. Fresh clone
cd ~/tt
rm -rf tt-metal
git clone --recurse-submodules git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git lfs pull
git submodule foreach --recursive "git lfs pull"

# 2. Install dependencies (requires sudo)
sudo ./install_dependencies.sh

# 3. Create Python environment
./create_venv.sh

# 4. Build tt-metal (required for tt-train dependencies)
source python_env/bin/activate
./build_metal.sh --debug --build-all --enable-ccache
deactivate
```

#### Step 2: Get Bug Reproduction Tests

The bug reproduction tests are available in the bug report branch:

```bash
# Fetch and cherry-pick the bug reproduction tests
git fetch origin ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3
git cherry-pick 16165972af  # "[tt-train] Add bug reproduction tests for LayerNorm backward accumulation bug"
```

**Alternative:** View the test source code directly at:
https://github.com/tenstorrent/tt-metal/blob/ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp

#### Step 3: Build and Run Tests

```bash
# Standalone tt-train build (RECOMMENDED for consistency)
cd tt-train
rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug --clean-first

# Run bug reproduction tests (all should FAIL with max_diff ~1000)
./build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.BugRepro_*"
```

**Expected output:**
```
[ RUN      ] LayerNormBackwardOpTest.BugRepro_Deterministic_256Tiles
dx FAILED: max_diff=1000, mean_diff=1004.21
[  FAILED  ] LayerNormBackwardOpTest.BugRepro_Deterministic_256Tiles
...
7 FAILED TESTS
```

### Container Reproduction (If Using Docker)

When running in Docker containers, **proper device mounts are required**:

```bash
# CORRECT - Full device access (tests work correctly)
docker run --rm \
    --device=/dev/tenstorrent:/dev/tenstorrent \
    -v /dev:/dev \
    -v /sys:/sys \
    -v /dev/hugepages:/dev/hugepages \
    <image> <command>

# INCORRECT - Minimal mounts (tests fail for wrong reasons)
docker run --rm \
    --device=/dev/tenstorrent:/dev/tenstorrent \
    <image> <command>
```

**Without `-v /dev:/dev -v /sys:/sys`, tests fail due to container misconfiguration, not the kernel bug.**

### Original NIGHTLY Test (May Pass Due to Random Data)

```bash
# This test uses random data and may pass (bug is masked)
./build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit"
```

**Note:** This test passes on properly configured systems because random mean-zero data masks the accumulation bug.

## Test Results Summary

### Verified on Fresh Main (8321610e95, Dec 17, 2025)

**Build:** Native (non-containerized), standalone tt-train build (`tt-train/build/`)
**Machine:** movsianikov-tt (Wormhole n150 L)

| Test | Result | Notes |
|------|--------|-------|
| BugRepro_Deterministic_256Tiles | **FAIL** | max_diff=1000, mean_diff=1004.21 |
| BugRepro_Deterministic_128Tiles | **FAIL** | max_diff=1000, mean_diff=994.62 |
| BugRepro_Deterministic_DifferentValues | **FAIL** | max_diff=1000, mean_diff=1004.21 |
| BugRepro_Deterministic_8462Features | **FAIL** | max_diff=988, mean_diff=1016.30 |
| BugRepro_Deterministic_2048Features | **FAIL** | max_diff=1000, mean_diff=997.24 |
| BugRepro_TightTolerance_8462Features | **FAIL** | |
| BugRepro_TightTolerance_8192Features | **FAIL** | |
| NIGHTLY test (random data, atol=0.5) | **PASS** | Bug masked by mean-zero data |

**Key insight:** The NIGHTLY test passes, but the kernel is still mathematically incorrect. Deterministic tests prove this conclusively.

## Bug Reproduction Tests Added

Seven new tests added to `tt-train/tests/ops/layernorm_bw_fused_op_test.cpp`.

**Source code:** https://github.com/tenstorrent/tt-metal/blob/ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp

**Note:** These tests are in the bug report branch. See "Step 2: Get Bug Reproduction Tests" above for how to apply them to main.

### Deterministic Tests (Constant Inputs)
- `BugRepro_Deterministic_256Tiles` - 8192 features (256 tiles)
- `BugRepro_Deterministic_128Tiles` - 4096 features (128 tiles)
- `BugRepro_Deterministic_DifferentValues` - Different constant values
- `BugRepro_Deterministic_8462Features` - Matches NIGHTLY test parameters
- `BugRepro_Deterministic_2048Features` - 2048 features (64 tiles)

### Tight Tolerance Tests (Random Inputs)
- `BugRepro_TightTolerance_8462Features` - Random data, atol=0.01
- `BugRepro_TightTolerance_8192Features` - Random data, atol=0.01

**Test design:**
- Deterministic tests use constant positive values (dy=1.0, gamma=1.0, x=0.5)
- Tight tolerance (atol=0.01) catches accumulation errors
- These tests will FAIL until the kernel is fixed

## Technical Details

### Affected Code Path

- **File:** `tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp`
- **Function:** `compute_dy_gamma_sum()`
- **Trigger:** Feature dimension doesn't fit in L1 (uses block-based path)
- **Threshold:** Approximately >1024 features (depends on other buffer sizes)

### Implementation Architecture

```
ttml::metal::ops::layernorm_bw::LayerNormBackwardOperation::invoke()
  → ttnn::prim::ttml_layernorm_bw()  [registered in tt-train via ttnn::register_operation]
    → LayerNormBackwardDeviceOperation  [tt-train/sources/ttml/metal/ops/layernorm_bw/device/]
      → LayerNormBackwardProgramFactory  [tt-train/sources/ttml/metal/ops/layernorm_bw/device/]
        → layernorm_bw_kernel.cpp (BUGGY)  [tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/]
```

**Code ownership:** The kernel and device operation are implemented in **tt-train** (`tt-train/sources/ttml/`), but registered as a TTNN primitive (`ttnn::prim::ttml_layernorm_bw`) using TTNN's `ttnn::register_operation<>` infrastructure from tt-metal. The bug is in tt-train's kernel implementation.

### Two Code Paths

| Path | Condition | Status |
|------|-----------|--------|
| `EVERYTHING_FITS_IN_L1` | All row data fits in L1 | Likely correct |
| Block-based | Data too large for L1 | **BUGGY** |

## Historical Investigation Notes

### Earlier Flakiness Analysis (Dec 16)

Initial investigation showed the NIGHTLY test was flaky (0-20% pass rate). This was due to:
1. Random data sometimes producing values within tolerance
2. **Container misconfiguration** (missing `/dev` and `/sys` mounts) causing failures for wrong reasons

### Container Configuration Discovery (Dec 17)

Testing revealed that container configuration significantly affects test behavior:
- Without `-v /dev:/dev -v /sys:/sys`: Tests fail (container issue)
- With proper mounts: NIGHTLY test passes, deterministic tests fail (kernel bug)

This explains the apparent "flakiness" - it was partially due to inconsistent test environments.

## Recommended Fix

### Option 1: Add Proper Accumulation (Minimal Change)

```cpp
// In compute_dy_gamma_sum(), accumulate each tile's result
for (uint32_t col = 0; col < Wt; ++col) {
    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, working_register);
    if (col == 0) {
        copy_tile(working_register, sum_register);
    } else {
        add_binary_tile(sum_register, working_register, sum_register);
    }
}
```

### Option 2: Follow TTNN Pattern (Recommended)

Use circular buffer for accumulator persistence across loop iterations:
1. Add `cb_accumulate` circular buffer
2. Load previous accumulator at start of each iteration
3. Use `reduce_tile` with accumulation mode
4. Save accumulator at end of each iteration

## Environment

- **Device:** Wormhole n150 L (single card)
- **OS:** Ubuntu 22.04.5 LTS
- **Kernel:** 6.8.0-87-generic
- **tt-metal:** Fresh main (`8321610e95`, Dec 17, 2025)
- **Build:** Standalone tt-train (`tt-train/build/`)
- **Machine:** movsianikov-tt

## Files

**Repository Ownership:** All source files with the bug belong to the **tt-train** subproject within the tt-metal repository (`github.com/tenstorrent/tt-metal`). The `tt-train/` directory is NOT a git submodule or external dependency - it is a subdirectory within tt-metal, tracked directly in tt-metal's git history.

### Files in Main tt-metal Repository (where the bug exists)

| File | Path | Description |
|------|------|-------------|
| **Bug location** | [`tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp) | Compute kernel with accumulation bug |
| **Device operation** | [`tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_device_operation.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_device_operation.hpp) | TTNN operation registration |
| **Program factory** | [`tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.cpp) | Kernel program setup |
| **High-level wrapper** | [`tt-train/sources/ttml/metal/ops/layernorm_bw/layernorm_bw.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/layernorm_bw.cpp) | TTML operation entry point |

### Files in Bug Report Branch (bug reproduction tests)

| File | URL | Description |
|------|-----|-------------|
| **Test file** | [tt-train/tests/ops/layernorm_bw_fused_op_test.cpp](https://github.com/tenstorrent/tt-metal/blob/ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp) | Bug reproduction tests (BugRepro_* tests) |
| **This bug report** | [bug_report_layernorm_bw_nightly_failure.md](https://github.com/tenstorrent/tt-metal/blob/ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3/bug_report_layernorm_bw_nightly_failure.md) | This document |

- **Branch:** `ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3`
- **Branch URL:** https://github.com/tenstorrent/tt-metal/tree/ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3

## Reporter

- **Date:** 2025-12-16 (initial), 2025-12-17 (fresh main verification)
- **Machine:** movsianikov-tt
