# [ttnn/ops]: `ttnn::untilize` corrupts data on Blackhole, causing argmax and other operations to fail

## Issue Metadata

**Component / Area:** ops, kernels, data movement

**Issue Type:** Incorrect output / Data corruption

## Issue Description

### Observed

**Root Cause Identified: `ttnn::untilize` corrupts tensor data on Blackhole.**

The `TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask` test fails because `ttnn::untilize` produces corrupted output. Debug testing shows the data corruption pattern:

**Input tensor (TILE layout) with sequential values:**
```
Row 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...
Row 1: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, ...
Row 2: 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, ...
Row 3: 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, ...
```

**After `ttnn::untilize` (ROW_MAJOR layout) - CORRUPTED:**
```
Row 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...           (CORRECT)
Row 1: 100, 101, 102, 103, 104, 105, 106, 107, ... (CORRECT)
Row 2: 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, ... 260, 260, 262, 264, 264 (CORRUPTED)
Row 3: 300, 300, 302, 304, 304, 304, 306, 308, 308, 308, ... (CORRUPTED - odd indices skipped, even duplicated)
```

**Corruption pattern:** Odd indices are skipped and even indices are duplicated in later rows. This occurs even for aligned dimensions (width=64), not just unaligned (width=65).

This causes downstream operations like `argmax` to return garbage values because they operate on corrupted data.

### Expected

`ttnn::untilize` should preserve tensor data exactly when converting from TILE to ROW_MAJOR layout, as it does on Wormhole hardware.

## Steps to Reproduce the Issue

### 1. Steps (exact commands)

```bash
# Build tt-train (from tt-metal root)
./build_metal.sh -b Release --build-tt-train

# Or standalone tt-train build
cd $TT_METAL_HOME/tt-train
cmake -DCMAKE_BUILD_TYPE=Debug -B build -GNinja
cmake --build build --config Debug

# Run the focused untilize tests (minimal reproduction)
cd $TT_METAL_HOME
./tt-train/build/tests/ttml_tests --gtest_filter="DebugUntilizeTest.UntilizeOnly*"

# Run the original failing test
./tt-train/build/tests/ttml_tests --gtest_filter="TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask"
```

### 2. Input data / link or description

**Minimal reproduction test** in `tt-train/tests/ttnn_fixed/debug_untilize_test.cpp`:

```cpp
TEST_F(DebugUntilizeTest, UntilizeOnly65) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Shape {1, 1, 4, 65}
    xt::xarray<float>::shape_type shape = {1, 1, 4, 65};
    xt::xarray<float> a = xt::zeros<float>(shape);

    // Set known sequential values
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 65; ++col) {
            a(0, 0, row, col) = static_cast<float>(row * 100 + col);
        }
    }

    auto tensor_a = ttml::core::from_xtensor(a, device);  // Creates TILE layout
    auto untilized = ttnn::untilize(tensor_a);            // CORRUPTS DATA on Blackhole
    auto vec = ttml::core::to_vector(untilized);

    // Check data integrity - FAILS on Blackhole
    for (size_t row = 0; row < 4; ++row) {
        for (size_t col = 0; col < 65; ++col) {
            float expected = static_cast<float>(row * 100 + col);
            EXPECT_NEAR(vec[row * 65 + col], expected, 0.1f);
        }
    }
}
```

**Original failing test** in `tt-train/tests/ttnn_fixed/trivial_ttnn_ops_test.cpp:277-298`

### 3. Frequency

**Always** - 100% reproducible on Blackhole hardware.

The test passes consistently on Wormhole hardware.

## System Details

### 1. Software Versions

- **OS version:** Ubuntu 22.04.5 LTS
- **Kernel:** 5.15.0-164-generic
- **tt-metal commit:** `403df4beb0f31a2b771349e58a02a98d859b9039` (main branch)
- **Firmware bundle version:** 19.3.0
- **ETH FW version:** 1.7.1
- **KMD version:** 2.6.1
- **tt-smi version:** 3.0.39

### 2. Hardware Details

- **Product:** Blackhole
- **Card/System:** P150 (PCI device 1e52:b140)
- **Single device configuration**

## Regression Info (Optional)

### Is this a regression?

Unknown - this may be a case of the operation never being validated on Blackhole rather than a regression.

### Regression Details

- **First bad version:** Unknown
- **Last known good version:** Works on Wormhole (N150/N300)
- **Git bisect status:** Not performed

## Logs & Diagnostics (Optional)

### Full test output (single test)

```
Running main() from gmock_main.cc
Note: Google Test filter = TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from TrivialTnnFixedTest
[ RUN      ] TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask
2025-12-15 16:33:42.809 | info     |             UMD | Starting topology discovery.
2025-12-15 16:33:42.895 | warning  |             UMD | Firmware bundle version 19.3.0 on the system is newer than the maximum supported version 19.1.0 for blackhole architecture. New features may not be supported.
...
/home/ivoitovych/tt/tt-metal/tt-train/tests/ttnn_fixed/trivial_ttnn_ops_test.cpp:296: Failure
Expected: (v) < (64), actual: 3201515335 vs 64
Expected: (v) < (64), actual: 3217342221 vs 64
Expected: (v) < (64), actual: 1066712917 vs 64
Expected: (v) < (64), actual: 3200859828 vs 64
Expected: (v) < (64), actual: 3202563797 vs 64
Expected: (v) < (64), actual: 3207970364 vs 64
Expected: (v) < (64), actual: 3217375136 vs 64
Expected: (v) < (64), actual: 3204431830 vs 64
Expected: (v) < (64), actual: 1072873412 vs 64
Expected: (v) < (64), actual: 3207184383 vs 64
Expected: (v) < (64), actual: 1058192921 vs 64
[  FAILED  ] TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask (1548 ms)
```

### Test group summary

```
[==========] 11 tests from TrivialTnnFixedTest ran.
[  PASSED  ] 10 tests.
[  FAILED  ] 1 test: TestSamplingPositiveTemperatureWithMask
```

### Full ttml_tests summary

```
[==========] 323 tests from 74 test suites ran. (344005 ms total)
[  PASSED  ] 274 tests.
[  SKIPPED ] 48 tests (N300 multi-device tests, NIGHTLY tests)
[  FAILED  ] 1 test: TrivialTnnFixedTest.TestSamplingPositiveTemperatureWithMask
```

## Impact & Priority (Optional)

### Priority

**P2** - Medium

### Impact

- **Affected workflows:** ANY operation using `ttnn::untilize` on Blackhole (TILE→ROW_MAJOR conversion), including but not limited to: sampling, argmax after tiled ops, data inspection/export
- **Affected users:** All developers using ttnn operations on Blackhole hardware
- **Release or date risk:** Blocks Blackhole validation for any workflow requiring untilize

## Analysis Notes

### Root Cause Identified

**The bug is in `ttnn::untilize` on Blackhole, NOT in argmax or the mask operation.**

Debug testing confirmed:
1. `ArgmaxUnaligned65NoUntilize` - **PASSES** - argmax works correctly when data is already in ROW_MAJOR layout
2. `UntilizeOnly64` - **FAILS** - even aligned dimensions show corruption in later rows
3. `UntilizeOnly65` - **FAILS** - data corruption in rows 2+
4. `UntilizeOnly33` - **FAILS** - data corruption in rows 2+

### Data Corruption Pattern

The corruption shows a specific pattern suggesting stride/address calculation errors:
- **Odd indices are skipped**
- **Even indices are duplicated**
- **First 1-2 rows are often correct, later rows are corrupted**

Example (width=64, row 3):
```
Expected: 300, 301, 302, 303, 304, 305, 306, 307, 308, 309
Actual:   300, 300, 302, 304, 304, 304, 306, 308, 308, 308
```

### Likely Root Cause

The pattern (reading every other value twice) suggests:
1. **Incorrect face/subtile width calculation** for Blackhole architecture
2. **Address stride error** in the untilize kernel when reading TILE format data
3. **Possible 16-bit vs 32-bit stride confusion** in data movement

### Files to Investigate

- `ttnn/cpp/ttnn/operations/data_movement/untilize/device/` - untilize kernel implementation
- Blackhole-specific TILE layout parameters vs Wormhole
- Face width and subtile dimensions for Blackhole architecture
