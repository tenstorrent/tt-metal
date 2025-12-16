# Bug Report: LayerNorm Backward NIGHTLY Test Failure

## Summary

The NIGHTLY test `LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit` fails with numerical precision errors on Wormhole n150.

## Test Information

- **Test Name:** `LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit`
- **Test File:** `tt-train/tests/ops/layernorm_bw_fused_op_test.cpp:201-203`
- **Test Parameters:** `CompareKernelVsXArray(3, 273, 1, 8462)`
  - batch_size = 3
  - height = 273
  - width = 1
  - features = 8462
- **Test Category:** NIGHTLY (not post-commit)

## Error Details

```
/home/ivoitovych/tt/tt-metal/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp:183: Failure
Value of: xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F)
  Actual: false
Expected: true
```

The test fails 3 times (likely for dx, dgamma, and dbeta comparisons) with tolerances:
- rtol = 1.0e-3 (relative tolerance)
- atol = 5e-1 (absolute tolerance = 0.5)

Note: The 0.5 absolute tolerance is already quite relaxed, yet the test still fails.

## Environment

### Hardware
- **Device:** Wormhole n150 L (single card)
- **Board Number:** 100018611902024
- **PCI Device ID:** 0

### Software
- **OS:** Ubuntu 22.04.5 LTS
- **Kernel:** 6.8.0-87-generic
- **tt-metal version:** v0.66.0-dev20251214-34-g228e5783f0
- **Build type:** Debug (standalone tt-train build)

### Build Commands
```bash
cd tt-train
rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug
```

## Test History

- **Test Added:** PR #30389 "[tt-train] LayerNorm forward + backward"
- **Author:** Daniil Yurshevich
- **Date:** 2025-11-21
- **Commit:** f44d93de69f

## Related Tests

Other LayerNorm backward tests in the same suite **PASS**:
- `MetalLayerNormBw_OneTile` - PASSED
- `MetalLayerNormBw_TwoIncompleteTiles` - PASSED
- `MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4` - PASSED (32043 ms)
- `MetalLayerNormBw_OneTilePerRow` - PASSED

The key difference is the failing test has:
- Large feature dimension (8462) that doesn't fit in L1
- The test name explicitly mentions "NoL1Fit"

## Hypothesis

The layer norm backward kernel may have numerical precision issues when:
1. Feature dimension is very large (8462 features)
2. Data doesn't fit in L1 cache, requiring different memory access patterns

The similar test `MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4` with features=8191 passes, suggesting the issue may be specific to certain feature dimensions or the combination of parameters.

## Reproduction Steps

```bash
# Build tt-train
cd $TT_METAL_HOME/tt-train
cmake -DCMAKE_BUILD_TYPE=Debug -B build -GNinja
cmake --build build

# Run failing test
./build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit"
```

## Full Test Output

```
[ RUN      ] LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit
2025-12-16 13:55:34.215 | info     |           Metal | DPRINT Server detached device 0
2025-12-16 13:55:34.225 | info     |     Distributed | Using auto discovery to generate mesh graph.
2025-12-16 13:55:34.225 | info     |     Distributed | Constructing control plane using auto-discovery
2025-12-16 13:55:34.231 | info     |           Metal | DPRINT enabled on device 0, worker core (x=0,y=0)
2025-12-16 13:55:34.231 | info     |           Metal | DPRINT Server attached device 0
/home/ivoitovych/tt/tt-metal/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp:183: Failure
Value of: xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F)
  Actual: false
Expected: true
/home/ivoitovych/tt/tt-metal/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp:183: Failure
Value of: xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F)
  Actual: false
Expected: true
/home/ivoitovych/tt/tt-metal/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp:183: Failure
Value of: xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F)
  Actual: false
Expected: true
[  FAILED  ] LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit (27002 ms)
```

## Test Suite Summary

```
[==========] 444 tests from 84 test suites ran. (1763910 ms total)
[  PASSED  ] 404 tests.
[  SKIPPED ] 39 tests (N300 multi-device tests)
[  FAILED  ] 1 test
```

## Suggested Investigation

1. Add debug output to compare actual vs expected values for dx, dgamma, dbeta
2. Check if the issue is related to specific value ranges or NaN/Inf
3. Compare memory access patterns between passing and failing tests
4. Verify the layer norm backward kernel handles non-L1-fitting data correctly

## Reproduction Confirmed

**Date:** 2025-12-16 15:41 UTC

Bug successfully reproduced on clean rebuild:

```bash
# Clean rebuild
cd $TT_METAL_HOME/tt-train
rm -rf build/
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build -GNinja
cmake --build build --config Debug

# Run test
./build/tests/ttml_tests --gtest_filter="LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit"
```

**Output:**
```
[ RUN      ] LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit
2025-12-16 15:41:20.845 | info     |           Metal | DPRINT enabled on device 0, worker core (x=0,y=0) (virtual (x=18,y=18)). (dprint_server.cpp:688)
2025-12-16 15:41:20.845 | info     |           Metal | DPRINT Server attached device 0 (dprint_server.cpp:735)
/home/ivoitovych/tt/tt-metal/tt-train/tests/ops/layernorm_bw_fused_op_test.cpp:183: Failure
Value of: xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, 5e-1F)
  Actual: false
Expected: true
[  FAILED  ] LayerNormBackwardOpTest.NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit (27633 ms)
```

**Observations:**
- Bug is consistently reproducible on clean builds
- This run showed 1 failure (dx only), vs 3 failures in original run (dx, dgamma, dbeta)
- The variation suggests the failures may be data-dependent (random test data each iteration)
- Test duration: ~27 seconds (consistent with original)

## Reporter

- **Date:** 2025-12-16
- **Machine:** movsianikov-tt
