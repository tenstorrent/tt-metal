# tanh_bw Baseline ULP Report

**Date:** 2026-03-07
**tt-metal commit:** `1c22985943` (main)
**Hardware:** Wormhole n150 L (movsianikov-tt)
**Issue:** [#35885](https://github.com/tenstorrent/tt-metal/issues/35885)
**Test file:** `tests/ttnn/unit_tests/gtests/test_tanh_bw_ulp.cpp`

## Summary

`ttnn::tanh_bw` produces **Max ULP = 15,139** across the BF16 range. The bug is caused by catastrophic cancellation in the composite formula `1 - tanh²(x)`: when `tanh(x)` saturates to ±1.0 in BF16 for |x| > ~3.4, `tanh² = 1.0` exactly, so `1 - 1 = 0`. The true values (e.g., sech²(4) = 0.0013) are perfectly representable in BF16.

For comparison, forward `ttnn::tanh` achieves **Max ULP = 1** using a polynomial approximation.

## Per-Segment ULP Analysis (DAZ+FTZ Model)

All ~65,026 valid BF16 values swept with grad = 1.0 (tests sech²(x) directly).

| Region | Count | Mean ULP | Max ULP | Worst x | Status |
|--------|------:|--------:|---------:|--------:|--------|
| x < -10 | 15,967 | 136.48 | 12,666 | -10.0625 | FAIL |
| [-10, -5) | 128 | 13,738.42 | 14,515 | -5.0312 | FAIL |
| [-5, -4) | 32 | 14,709.56 | 14,886 | -4.0312 | FAIL |
| [-4, -3) | 64 | 10,106.89 | **15,139** | **-3.3438** | FAIL |
| [-3, -2) | 64 | 22.30 | 95 | -3.0000 | FAIL |
| [-2, -1) | 128 | 3.67 | 17 | -1.7734 | FAIL |
| [-1, 0) | 16,129 | 0.09 | 3 | -0.8984 | FAIL |
| x == 0 | 2 | 0.00 | 0 | 0.0000 | pass |
| (0, 1) | 16,128 | 0.09 | 3 | 0.8984 | FAIL |
| [1, 2) | 128 | 3.68 | 17 | 1.7734 | FAIL |
| [2, 3) | 64 | 20.81 | 90 | 2.9844 | FAIL |
| [3, 4) | 64 | 9,875.62 | **15,139** | **3.3438** | FAIL |
| [4, 5) | 32 | 14,721.09 | 14,896 | 4.0000 | FAIL |
| [5, 10) | 128 | 13,752.80 | 14,527 | 5.0000 | FAIL |
| x >= 10 | 15,968 | 137.27 | 12,686 | 10.0000 | FAIL |
| **OVERALL** | **65,026** | **155.59** | **15,139** | **3.3438** | **FAIL** |

## Cumulative ULP Distribution

| ULP <= | Count | Percent |
|-------:|------:|--------:|
| 0 | 60,866 | 93.60% |
| 1 | 63,708 | 97.97% |
| 2 | 63,796 | 98.11% |
| 3 | 63,838 | 98.17% |
| 5 | 63,876 | 98.23% |
| 10 | 63,936 | 98.32% |
| 50 | 64,042 | 98.49% |
| 100 | 64,074 | 98.54% |
| 1,000 | 64,096 | 98.57% |
| 15,000 | 64,976 | 99.92% |

**952 values** (1.46%) have ULP > 100 — these are the catastrophic failures in the saturation region.

## Saturation Region Detail

The |x| > 3 region where `tanh(x)` saturates to ±1.0 in BF16:

| x | Expected (sech²) | Actual | ULP | Note |
|----:|---------:|-------:|------:|------|
| -5.0000 | 1.812e-04 | 0.0 | 14,527 | Representable in BF16, returned as 0 |
| -4.5000 | 4.921e-04 | 0.0 | 14,722 | " |
| -4.0000 | 1.335e-03 | 0.0 | 14,896 | " |
| -3.7500 | 2.197e-03 | 0.0 | 14,993 | " |
| -3.5000 | 3.632e-03 | 0.0 | 15,087 | " |
| **-3.3438** | **4.944e-03** | **0.0** | **15,139** | **Worst value in entire BF16 range** |
| -3.0000 | 9.827e-03 | 1.5625e-02 | 95 | Non-zero but wrong |
| -2.5000 | 2.649e-02 | 2.344e-02 | 25 | Non-zero but wrong |
| 2.5000 | 2.649e-02 | 2.344e-02 | 25 | Symmetric |
| 3.0000 | 9.827e-03 | 1.5625e-02 | 95 | " |
| **3.3438** | **4.944e-03** | **0.0** | **15,139** | **Worst value (symmetric)** |
| 3.5000 | 3.632e-03 | 0.0 | 15,087 | " |
| 4.0000 | 1.335e-03 | 0.0 | 14,896 | " |
| 5.0000 | 1.812e-04 | 0.0 | 14,527 | " |

## Root Cause

Source: `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` (lines 289-313)

```cpp
Tensor tanh_res = ttnn::tanh(input);      // saturates to ±1.0 for |x| > ~3.4
tanh_res = ttnn::square(tanh_res);        // 1.0² = 1.0
tanh_res = ttnn::rsub(tanh_res, 1.0f);   // 1.0 - 1.0 = 0.0  ← BUG
grad_tensor = ttnn::multiply(grad, tanh_res);  // grad * 0 = 0
```

The SFPU kernel `ckernel_sfpu_tanh_derivative.h` has the same bug at line 29:
```cpp
val = val * (-val) + vConst1;  // 1 - tanh(x)² — same cancellation
```

## Test Results

9 tests total, 4 pass / 5 fail on baseline:

| Test | Result | What it tests |
|------|--------|---------------|
| DerivativeAtZero | PASS | sech²(0) = 1 |
| DerivativeNearZero | PASS | Small |x| where tanh doesn't saturate |
| WithGradientScaling | PASS | grad * sech²(x) for small |x| |
| ReferenceImplementationVerification | PASS | fp64 golden reference correctness |
| DerivativeAtPositiveValues | **FAIL** | x=3: ULP=95, x=5: ULP=14,527 |
| DerivativeAtNegativeValues | **FAIL** | x=-3: ULP=95, x=-4: ULP=14,896 |
| PerSegmentULPAnalysis | **FAIL** | 14 of 15 regions exceed ULP threshold |
| CumulativeULPDistribution | **FAIL** | Max ULP 15,139, only 98.11% within 2 ULP |
| SaturationRegionAnalysis | **FAIL** | All |x| > 3 values return wrong results |

## Fix Target

Replace the composite `1 - tanh²(x)` with a fused SFPU kernel using piecewise polynomials for sech²(x), following the same pattern as the gelu_bw fix (PR #39303). Target: **Max ULP <= 2** across the entire BF16 range.
