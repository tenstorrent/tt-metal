# tanh_bw Baseline ULP Report

**Date:** 2026-03-07
**tt-metal commit:** `1c22985943` (main)
**Hardware:** Wormhole n150 L (movsianikov-tt)
**Issue:** [#35885](https://github.com/tenstorrent/tt-metal/issues/35885)
**Test files:** `tests/ttnn/unit_tests/gtests/test_tanh_bw_ulp.cpp`, `tests/ttnn/unit_tests/gtests/test_tanh_fw_ulp.cpp`
**ULP model:** DAZ+FTZ, round-to-nearest-even (RNE) for fp64-to-BF16 reference conversion

## Summary

`ttnn::tanh_bw` produces **Max ULP = 15,140** across the BF16 range due to catastrophic cancellation in `1 - tanh²(x)`. When `tanh(x)` saturates to ±1.0 in BF16 for |x| > ~3.4, `tanh² = 1.0` exactly, so `1 - 1 = 0`. The true values (e.g., sech²(4) = 0.0013) are perfectly representable in BF16.

For comparison, forward `ttnn::tanh` achieves **Max ULP = 1** (see Reference section below).

## tanh_bw Per-Segment ULP Analysis

All ~65,026 valid BF16 values swept with grad = 1.0 (tests sech²(x) directly).

| Region | Count | Mean ULP | Max ULP | Worst x | Status |
|--------|------:|--------:|---------:|--------:|--------|
| x < -10 | 15,967 | 136.49 | 12,667 | -10.0625 | FAIL |
| [-10, -5) | 128 | 13,738.99 | 14,516 | -5.0312 | FAIL |
| [-5, -4) | 32 | 14,710.03 | 14,886 | -4.0312 | FAIL |
| [-4, -3) | 64 | 10,107.06 | **15,140** | **-3.3438** | FAIL |
| [-3, -2) | 64 | 22.23 | 94 | -3.0000 | FAIL |
| [-2, -1) | 128 | 3.41 | 17 | -1.7734 | FAIL |
| [-1, 0) | 16,129 | 0.01 | 3 | -0.9336 | FAIL |
| x == 0 | 2 | 0.00 | 0 | 0.0000 | pass |
| (0, 1) | 16,128 | 0.01 | 3 | 0.9336 | FAIL |
| [1, 2) | 128 | 3.41 | 17 | 1.7734 | FAIL |
| [2, 3) | 64 | 20.78 | 89 | 2.9844 | FAIL |
| [3, 4) | 64 | 9,875.77 | **15,140** | **3.3438** | FAIL |
| [4, 5) | 32 | 14,721.59 | 14,897 | 4.0000 | FAIL |
| [5, 10) | 128 | 13,753.37 | 14,527 | 5.0000 | FAIL |
| x >= 10 | 15,968 | 137.28 | 12,687 | 10.0000 | FAIL |
| **OVERALL** | **65,026** | **155.55** | **15,140** | **3.3438** | **FAIL** |

## tanh_bw Cumulative ULP Distribution

| ULP <= | Count | Percent |
|-------:|------:|--------:|
| 0 | 63,380 | 97.47% |
| 1 | 63,760 | 98.05% |
| 2 | 63,810 | 98.13% |
| 3 | 63,840 | 98.18% |
| 5 | 63,888 | 98.25% |
| 10 | 63,942 | 98.33% |
| 50 | 64,042 | 98.49% |
| 100 | 64,074 | 98.54% |
| 1,000 | 64,096 | 98.57% |
| 15,000 | 64,976 | 99.92% |

**952 values** (1.46%) have ULP > 100 — catastrophic failures in the saturation region.

## Saturation Region Detail

The |x| > 3 region where `tanh(x)` saturates to ±1.0 in BF16:

| x | Expected (sech²) | Actual | ULP | Note |
|----:|---------:|-------:|------:|------|
| -5.0000 | 1.812e-04 | 0.0 | 14,527 | Representable in BF16, returned as 0 |
| -4.5000 | 4.921e-04 | 0.0 | 14,722 | " |
| -4.0000 | 1.335e-03 | 0.0 | 14,897 | " |
| -3.7500 | 2.197e-03 | 0.0 | 14,993 | " |
| -3.5000 | 3.632e-03 | 0.0 | 15,087 | " |
| **-3.3438** | **4.944e-03** | **0.0** | **15,140** | **Worst value in entire BF16 range** |
| -3.0000 | 9.827e-03 | 1.5625e-02 | 94 | Non-zero but wrong |
| -2.5000 | 2.649e-02 | 2.344e-02 | 25 | Non-zero but wrong |
| 2.5000 | 2.649e-02 | 2.344e-02 | 25 | Symmetric |
| 3.0000 | 9.827e-03 | 1.5625e-02 | 94 | " |
| **3.3438** | **4.944e-03** | **0.0** | **15,140** | **Worst value (symmetric)** |
| 3.5000 | 3.632e-03 | 0.0 | 15,087 | " |
| 4.0000 | 1.335e-03 | 0.0 | 14,897 | " |
| 5.0000 | 1.812e-04 | 0.0 | 14,527 | " |

## Root Cause

Source: `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` (lines 289-313)

```cpp
Tensor tanh_res = ttnn::tanh(input);      // saturates to ±1.0 for |x| > ~3.4
tanh_res = ttnn::square(tanh_res);        // 1.0² = 1.0
tanh_res = ttnn::rsub(tanh_res, 1.0f);   // 1.0 - 1.0 = 0.0  <- BUG
grad_tensor = ttnn::multiply(grad, tanh_res);  // grad * 0 = 0
```

The SFPU kernel `ckernel_sfpu_tanh_derivative.h` has the same bug at line 29:
```cpp
val = val * (-val) + vConst1;  // 1 - tanh(x)² — same cancellation
```

## tanh_bw Test Results

9 tests total, 4 pass / 5 fail on baseline:

| Test | Result | What it tests |
|------|--------|---------------|
| DerivativeAtZero | PASS | sech²(0) = 1 |
| DerivativeNearZero | PASS | Small |x| where tanh doesn't saturate |
| WithGradientScaling | PASS | grad * sech²(x) for small |x| |
| ReferenceImplementationVerification | PASS | fp64 golden reference correctness |
| DerivativeAtPositiveValues | **FAIL** | x=3: ULP=94, x=5: ULP=14,527 |
| DerivativeAtNegativeValues | **FAIL** | x=-3: ULP=94, x=-4: ULP=14,897 |
| PerSegmentULPAnalysis | **FAIL** | 14 of 15 regions exceed ULP threshold |
| CumulativeULPDistribution | **FAIL** | Max ULP 15,140, only 98.13% within 2 ULP |
| SaturationRegionAnalysis | **FAIL** | All |x| > 3 values return wrong results |

## Fix Target

Replace the composite `1 - tanh²(x)` with a fused SFPU kernel computing sech²(x) directly, following the tanh forward's infrastructure pattern. Target: **Max ULP <= 2** across the entire BF16 range.

---

## Implementation Research

### tanh Forward (Max ULP = 1) — How It Works

**Call chain:**
`ttnn::tanh(tensor)` → `Tanh::invoke()` → `detail::unary_impl({UnaryOpType::TANH})` → `prim::unary()` → `UnaryDeviceOperation` → `UnaryProgramFactory` → compute kernel `eltwise_sfpu.cpp` → macro expansion `tanh_tile<0u>(idst)` → LLK `llk_math_eltwise_unary_sfpu_tanh()` → `ckernel_sfpu_tanh.h`

**Key files:**

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | API registration, `Tanh` struct |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` | `Tanh::invoke()` — delegates to `UnaryOpType::TANH` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Macro generation: `tanh_tile_init<>()` / `tanh_tile<>(idst)` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel (macro substitution) |
| `tt_metal/hw/inc/api/compute/compute_kernel_api.h` | `tanh_tile_init<>()` / `tanh_tile<>()` templates |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_tanh.h` | LLK math wrapper |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h` | **SFPU kernel — the math** |

**Math (3 modes in `ckernel_sfpu_tanh.h`):**

1. **FP32 accurate (`_sfpu_tanh_fp32_accurate_`):**
   - |x| < 0.6: degree-8 minimax polynomial for tanh(x)/x (Sollya fpminimax, even powers)
   - |x| >= 0.6: `tanh(x) = 2·sigmoid(2x) - 1` (uses `_sfpu_sigmoid_` → `_sfpu_exp_improved_` → `_sfpu_reciprocal_`)

2. **BF16 polynomial (`_sfpu_tanh_polynomial_`):**
   - degree-6 minimax polynomial on |x|, clamped to [-1, 1], sign restored
   - Coefficients: c1=0.999, c2=0.031, c3=-0.489, c4=0.282, c5=-0.067, c6=0.006

3. **Approximate (LUT):** hardware `lut()` instruction with 3 programmed registers

### tanh_bw (Max ULP = 15,140) — Current Implementation

**The backward is a pure composite (host-side) decomposition — no device operation, no SFPU kernel.**

**Call chain:**
`ttnn::tanh_bw(grad, input)` → `ExecuteUnaryBackwardTanh::invoke()` → 4 separate tensor ops on host

**Buggy code** (`unary_backward.cpp` lines 289-303):
```cpp
Tensor tanh_res = ttnn::tanh(input, output_mem_config);        // saturates to ±1.0
tanh_res = ttnn::square(tanh_res, output_mem_config);           // 1.0² = 1.0
tanh_res = ttnn::rsub(tanh_res, 1.0f, ...);                    // 1.0 - 1.0 = 0.0 ← BUG
ttnn::multiply(grad, tanh_res, ..., input_grad);                // grad * 0 = 0
```

**Orphaned SFPU kernel** (`ckernel_sfpu_tanh_derivative.h`, 7 lines of math):
```cpp
val = lut(val, l0, l1, l2);           // compute tanh(x) via LUT
val = val * (-val) + vConst1;         // 1 - tanh²(x) — SAME BUG
```
This kernel exists but is **never called** by `ttnn::tanh_bw`. It has the same cancellation bug anyway.

**No enum dispatch:** Backward ops have no `UnaryBackwardOpType` enum — they are all standalone structs with `invoke()`.

### Fix Strategy

Follow the tanh forward pattern — replace the composite with a fused SFPU kernel:

1. **New SFPU kernel** in `ckernel_sfpu_tanh_derivative.h`: compute sech²(x) = 1/cosh²(x) directly using the same infrastructure as tanh forward (piecewise polynomial or `_sfpu_exp_` + reciprocal)
2. **Wire through existing unary infrastructure**: `UnaryOpType::TANH_DERIVATIVE` or a new backward device operation
3. **Two input tensors**: Unlike forward (1 input), backward needs grad and input — may require the gelu_bw experimental pattern (binary device operation) rather than simple unary

**Blueprint:** The experimental GELU backward device operation (`ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/`) shows how to create a fused backward op with a custom compute kernel taking 2 inputs.

---

## Reference: tanh Forward Baseline

`ttnn::tanh` (forward) achieves **Max ULP = 1**, proving correct BF16 precision is achievable.

| Region | Count | Mean ULP | Max ULP | Worst x | Status |
|--------|------:|--------:|---------:|--------:|--------|
| x < -10 | 15,967 | 0.00 | 0 | — | pass |
| [-10, -5) | 128 | 0.00 | 0 | — | pass |
| [-5, -4) | 32 | 0.00 | 0 | — | pass |
| [-4, -3) | 64 | 0.16 | 1 | -3.016 | pass |
| [-3, -2) | 64 | 0.22 | 1 | -2.047 | pass |
| [-2, -1) | 128 | 0.20 | 1 | -1.023 | pass |
| [-1, 0) | 16,129 | 0.00 | 1 | -1.175e-38 | pass |
| x == 0 | 2 | 0.00 | 0 | 0.000 | pass |
| (0, 1) | 16,128 | 0.00 | 1 | 1.175e-38 | pass |
| [1, 2) | 128 | 0.20 | 1 | 1.023 | pass |
| [2, 3) | 64 | 0.20 | 1 | 2.047 | pass |
| [3, 4) | 64 | 0.17 | 1 | 3.000 | pass |
| [4, 5) | 32 | 0.00 | 0 | — | pass |
| [5, 10) | 128 | 0.00 | 0 | — | pass |
| x >= 10 | 15,968 | 0.00 | 0 | — | pass |
| **OVERALL** | **65,026** | **0.00** | **1** | — | **pass** |

99.67% exact (ULP=0), 100% within 1 ULP. 7 tests, all pass.

### Forward vs Backward Comparison

| Metric | tanh (forward) | tanh_bw (backward) |
|--------|---------------:|-------------------:|
| Max ULP | **1** | **15,140** |
| Mean ULP | **0.00** | **155.55** |
| % exact (ULP=0) | **99.67%** | 97.47% |
| % within 1 ULP | **100%** | 98.05% |
| % within 2 ULP | **100%** | 98.13% |
| Values with ULP > 100 | 0 | **952** |
