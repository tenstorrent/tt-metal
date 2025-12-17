### Component / Area

tt-train, kernels, ops

### Issue Type (optional)

Bad Outputs

### Observed

The TTML LayerNorm backward kernel (`layernorm_bw_kernel.cpp`) has a critical accumulation bug in `compute_dy_gamma_sum()` that causes ~99% of data to be lost when processing tensors with large feature dimensions that don't fit in L1 cache.

**Root cause:** In the block-based code path, each loop iteration OVERWRITES `sum_register` instead of accumulating.

**Buggy code locations** (GitHub permalinks):
- [L129](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp#L129), [L133](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp#L133) — non-block path
- [L283](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp#L283), [L287](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp#L287) — block-based path

```cpp
// BUGGY: Each iteration OVERWRITES sum_register instead of accumulating
for (uint32_t col = 0; col < Wt; ++col) {
    mul_tiles_bcast_rows(cb_dL_out_idx, cb_gamma_idx, block_idx, block_idx, sum_register);  // OVERWRITES!
}
// Only the last tile uses add_binary_tile for accumulation
add_binary_tile(sum_register, working_register, sum_register);
```

**Result:** For `Wt=265` tiles (8462 features), only the last 2 tiles contribute to the sum. ~99% of data is lost.

**Why existing NIGHTLY test passes:** The test uses random uniform[-1,1] data with mean~0. With loose tolerance (atol=0.5), the buggy sum (still mean-zero) often stays within bounds. Deterministic tests with constant positive values expose the bug definitively.

### Expected

All tiles should be accumulated in the sum. The correct pattern (from TTNN):

```cpp
for (uint32_t wt = 0; wt < Wt; wt += blk) {
    if (wt > 0) {
        copy_tile(cb_accumulate, 0, dst0);  // Load previous accumulator
    }
    reduce_tile(..., dst0);  // Accumulates into dst0
    pack_tile(dst0, cb_accumulate);  // Save for next iteration
}
```

For deterministic inputs (dy=1.0, gamma=1.0, x=0.5) with 8192 features:
- **Expected:** sum ~ 8192
- **Actual:** sum ~ 64 (only last 2 tiles)
- **Error:** max_diff ~ 1000

### 1. Steps (exact commands)

#### Fastest Repro (For Existing Builds)

If tt-metal is already cloned and tt-train previously built on commit `8321610e95` or later:

```bash
# 1. Fetch and cherry-pick bug reproduction tests
git fetch origin ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3
git cherry-pick 16165972af

# 2. Rebuild tt-train (from tt-metal root)
cd tt-train
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build

# 3. Run tests (all should FAIL with max_diff ~1000)
./build/tests/ttml_tests --gtest_filter=LayerNormBackwardOpTest.BugRepro_*
```

For fresh clones, see full steps below.

#### Native Build (Recommended) — Full Steps

```bash
# 1. Fresh clone
cd ~/tt
git clone --recurse-submodules git@github.com:tenstorrent/tt-metal.git
cd tt-metal
git lfs pull
git submodule foreach --recursive "git lfs pull"

# 2. Install dependencies (requires sudo)
sudo ./install_dependencies.sh

# 3. Create Python environment
./create_venv.sh

# 4. Build tt-metal
source python_env/bin/activate
./build_metal.sh --debug --build-all --enable-ccache
deactivate

# 5. Fetch and apply bug reproduction tests
git fetch origin ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3
git cherry-pick 16165972af  # "[tt-train] Add bug reproduction tests for LayerNorm backward accumulation bug"

# 6. Standalone tt-train build
cd tt-train
rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug --clean-first

# 7. Run bug reproduction tests (all should FAIL)
./build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.BugRepro_*"
```

#### Container Reproduction (If Using Docker)

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

#### Original NIGHTLY Test (May Pass Due to Random Data)

```bash
# This test uses random data and may pass (bug is masked)
./build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit"
```

### 2. Input data / link or description

Bug reproduction tests use deterministic constant inputs:
- `dy = 1.0` (gradient)
- `gamma = 1.0` (scale parameter)
- `x = 0.5` (input tensor)
- Feature dimensions: 2048, 4096, 8192, 8462 (all trigger block-based path)

Bug reproduction tests were added in commit [16165972af](https://github.com/tenstorrent/tt-metal/commit/16165972af).

Test source code: https://github.com/tenstorrent/tt-metal/blob/ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp

### 3. Frequency

**Always** - 100% reproducible with deterministic tests.

The original NIGHTLY test (`NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit`) passes because random mean-zero data masks the bug.

| Test | Result | Notes |
|------|--------|-------|
| BugRepro_Deterministic_256Tiles | **FAIL** | max_diff=1000 |
| BugRepro_Deterministic_128Tiles | **FAIL** | max_diff=1000 |
| BugRepro_Deterministic_DifferentValues | **FAIL** | max_diff=1000 |
| BugRepro_Deterministic_8462Features | **FAIL** | max_diff=988 |
| BugRepro_Deterministic_2048Features | **FAIL** | max_diff=1000 |
| BugRepro_TightTolerance_8462Features | **FAIL** | |
| BugRepro_TightTolerance_8192Features | **FAIL** | |
| NIGHTLY test (random data, atol=0.5) | PASS | Bug masked |

### 1. Software Versions

- **OS version:** Ubuntu 22.04.5 LTS
- **Kernel:** 6.8.0-87-generic
- **tt-metal:** Fresh main (`8321610e95`, Dec 17, 2025)
- **Build:** Standalone tt-train (`tt-train/build/`)

### 2. Hardware Details

- **Product:** Wormhole
- **Card/System:** n150 L (single card)

### Is this a regression?

Unknown - the bug may have existed since the kernel was written.

### Regression Details

- **First bad version:** Unknown
- **Last known good version:** Unknown
- **Git bisect status:** Not performed for original introduction

### Logs & Diagnostics

```
[ RUN      ] LayerNormBackwardOpTest.BugRepro_Deterministic_256Tiles
dx FAILED: max_diff=1000, mean_diff=1004.21
[  FAILED  ] LayerNormBackwardOpTest.BugRepro_Deterministic_256Tiles (1523 ms)
[ RUN      ] LayerNormBackwardOpTest.BugRepro_Deterministic_128Tiles
dx FAILED: max_diff=1000, mean_diff=994.62
[  FAILED  ] LayerNormBackwardOpTest.BugRepro_Deterministic_128Tiles (1089 ms)
...
[  FAILED  ] 7 tests
```

### Priority

P2

### Impact

- **Affected workflows:** Any training workload using LayerNorm backward pass with feature dimensions > ~1024 (exact threshold depends on L1 availability)
- **Release or date risk:** The bug produces incorrect gradients, which can cause training to fail silently or produce suboptimal models. However, the effect may be masked by mean-zero gradient distributions in practice.
- **Development impact:** This bug causes flickery test suite failures in TTML/tt-train development. When verifying changes cause no regression, the flickering NIGHTLY test produces false alarms that distract the development process and require days of investigation to identify the root cause.

---

## Historical Investigation Notes

### Why Deterministic Tests Were Created

Initial investigation (Dec 16, 2025) showed the NIGHTLY test appeared flaky (0-20% pass rate). Further analysis revealed:

1. **Random data masks the bug:** The NIGHTLY test uses random uniform[-1,1] data with mean~0. Even with ~99% of tiles lost, the sum of mean-zero data is still ~0, often passing loose tolerance checks.

2. **Container misconfiguration:** Missing `/dev` and `/sys` mounts in Docker caused test failures for unrelated reasons, confusing the flakiness analysis.

3. **Solution:** Deterministic tests with constant positive values (dy=1.0, gamma=1.0, x=0.5) definitively expose the bug - expected sum ~8192 vs actual ~64.

This explains why the NIGHTLY test passes on properly configured systems while the kernel is still mathematically incorrect.

---

## Additional Technical Details

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

**Code ownership:** The kernel and device operation are implemented in **tt-train** (`tt-train/sources/ttml/`), but registered as a TTNN primitive (`ttnn::prim::ttml_layernorm_bw`) using TTNN's `ttnn::register_operation<>` infrastructure from tt-metal.

### Two Code Paths

| Path | Condition | Status |
|------|-----------|--------|
| `EVERYTHING_FITS_IN_L1` | All row data fits in L1 | Likely correct |
| Block-based | Data too large for L1 | **BUGGY** |

### Recommended Fix

**Option 1: Add Proper Accumulation (Minimal Change)**

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

**Option 2: Follow TTNN Pattern (Recommended)**

Use circular buffer for accumulator persistence across loop iterations:
1. Add `cb_accumulate` circular buffer
2. Load previous accumulator at start of each iteration
3. Use `reduce_tile` with accumulation mode
4. Save accumulator at end of each iteration

### Files

All source files belong to the **tt-train** subproject within the tt-metal repository (`github.com/tenstorrent/tt-metal`). The `tt-train/` directory is NOT a git submodule or external dependency - it is a subdirectory within tt-metal, tracked directly in tt-metal's git history.

| File | Description |
|------|-------------|
| [`layernorm_bw_kernel.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/kernels/compute/layernorm_bw_kernel.cpp) | Compute kernel with accumulation bug |
| [`layernorm_bw_device_operation.hpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_device_operation.hpp) | TTNN operation registration |
| [`layernorm_bw_program_factory.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/device/layernorm_bw_program_factory.cpp) | Kernel program setup |
| [`layernorm_bw.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt-train/sources/ttml/metal/ops/layernorm_bw/layernorm_bw.cpp) | High-level TTML operation wrapper |

### Acceptance Criteria

- [ ] All `LayerNormBackwardOpTest.BugRepro_*` tests PASS on Wormhole n150 L
- [ ] Original NIGHTLY test (`NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit`) still PASSES (no regression)
- [ ] No performance regression >5% on affected code paths (optional benchmark)

### Bug Report Branch

- **Branch:** `ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3`
- **URL:** https://github.com/tenstorrent/tt-metal/tree/ivoitovych/layernorm-bw-nightly-test-failure-bug-report-3
