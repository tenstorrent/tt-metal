# tanh Forward Baseline ULP Report

**Date:** 2026-03-07
**tt-metal commit:** `1c22985943` (main)
**Hardware:** Wormhole n150 L (movsianikov-tt)
**Test file:** `tests/ttnn/unit_tests/gtests/test_tanh_fw_ulp.cpp`

## Summary

`ttnn::tanh` (forward) achieves **Max ULP = 1** across all valid BF16 values, confirming that correct BF16 implementation of tanh-family functions is achievable. This serves as the baseline comparison for the tanh_bw fix.

## Per-Segment ULP Analysis (DAZ+FTZ Model)

All ~65,026 valid BF16 values swept (denormals excluded per DAZ policy).

| Region | Count | Mean ULP | Max ULP | Worst x | Status |
|--------|------:|--------:|---------:|--------:|--------|
| x < -10 | 15,967 | 0.00 | 0 | — | pass |
| [-10, -5) | 128 | 0.88 | 1 | -5.031 | pass |
| [-5, -4) | 32 | 1.00 | 1 | -4.031 | pass |
| [-4, -3) | 64 | 0.75 | 1 | -3.047 | pass |
| [-3, -2) | 64 | 0.45 | 1 | -2.062 | pass |
| [-2, -1) | 128 | 0.52 | 1 | -1.008 | pass |
| [-1, 0) | 16,129 | 0.08 | 1 | -1.175e-38 | pass |
| x == 0 | 2 | 0.00 | 0 | 0.000 | pass |
| (0, 1) | 16,128 | 0.08 | 1 | 1.175e-38 | pass |
| [1, 2) | 128 | 0.52 | 1 | 1.000 | pass |
| [2, 3) | 64 | 0.47 | 1 | 2.000 | pass |
| [3, 4) | 64 | 0.73 | 1 | 3.047 | pass |
| [4, 5) | 32 | 1.00 | 1 | 4.000 | pass |
| [5, 10) | 128 | 0.88 | 1 | 5.000 | pass |
| x >= 10 | 15,968 | 0.00 | 0 | — | pass |
| **OVERALL** | **65,026** | **0.05** | **1** | **1.175e-38** | **pass** |

## Cumulative ULP Distribution

| ULP <= | Count | Percent |
|-------:|------:|--------:|
| 0 | 61,944 | 95.26% |
| 1 | 65,026 | 100.00% |

**100%** of valid BF16 values are within 1 ULP.

## Comparison: Forward vs Backward

| Metric | tanh (forward) | tanh_bw (backward) |
|--------|---------------:|-------------------:|
| Max ULP | **1** | **15,139** |
| Mean ULP | **0.05** | **155.59** |
| % within 1 ULP | **100%** | 97.97% |
| % within 2 ULP | **100%** | 98.11% |
| Values with ULP > 100 | 0 | **952** |

This demonstrates that the backward implementation can and should match the forward's precision.

## Test Results

7 tests, all pass:

| Test | Result | What it tests |
|------|--------|---------------|
| ValueAtZero | PASS | tanh(0) = 0 exactly |
| PositiveValues | PASS | tanh(x) -> 1 for positive x, ULP <= 1 |
| NegativeValues | PASS | tanh(-x) = -tanh(x) symmetry, ULP <= 1 |
| NearZero | PASS | Small |x| region, ULP <= 2 (non-exact BF16 inputs) |
| PerSegmentULPAnalysis | PASS | All regions Max ULP <= 1 |
| CumulativeULPDistribution | PASS | 100% within 1 ULP |
| ReferenceImplementationVerification | PASS | fp64 reference correctness |
