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

**Blueprint:** The main API GELU backward device operation (PR #39303, `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/`) is the most directly applicable blueprint — same directory, same invoke() pattern, same CMakeLists.txt structure.

---

## Cross-Reference: Successful Fused Kernel Implementations

Three PRs demonstrate how to fix precision bugs by replacing composite decompositions with fused SFPU kernels. All three are open, authored by the same developer, and follow the same architectural pattern.

### PR #36366: Experimental GELU Backward (Max ULP 32,460 → 1)

**PR:** [#36366](https://github.com/tenstorrent/tt-metal/pull/36366) — `[TTNN] Fix experimental gelu_bw ULP errors: polynomial kernel replaces broken erf+exp (Max ULP 32,460 → 1)`
**State:** OPEN, REVIEW_REQUIRED
**Branch:** `origin/ivoitovych/issue-35971-gelu-bw-ulp-fix-pr`
**Changes:** +2773 / -107, 9 files

#### Architecture

The fix follows tt-metal's `DeviceOperationConcept` pattern. The existing device operation infrastructure was preserved — only the compute kernel was replaced.

**Files changed:**

| File | Change |
|------|--------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Added polynomial coefficients + evaluation functions |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Same (byte-identical pair) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h` | Added `gelu_derivative_tile()` API |
| `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/gelu_backward_program_factory.cpp` | Kernel selection: "none" → `eltwise_bw_gelu_poly.cpp` |
| `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_poly.cpp` | **NEW** — polynomial compute kernel |
| `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp` | **DELETED** — broken erf+exp kernel |
| `tests/ttnn/unit_tests/gtests/test_gelu_bw_ulp.cpp` | **NEW** — C++ ULP tests |
| `tests/ttnn/unit_tests/gtests/sources.cmake` | Registered test file |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp.py` | **NEW** — Python ULP tests |

#### Existing Device Operation (Preserved, Not New)

The experimental gelu_bw already had a binary device operation before the PR. The PR kept this infrastructure intact:

**Types** (`gelu_backward_device_operation_types.hpp`):
```cpp
struct GeluBackwardParams {
    const DataType output_dtype = DataType::INVALID;
    const MemoryConfig output_memory_config;
    const std::string approximate = "none";
};

struct GeluBackwardInputs {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;
};
```

**Device operation** (`gelu_backward_device_operation.hpp`):
```cpp
struct GeluBackwardDeviceOperation {
    using operation_attributes_t = GeluBackwardParams;
    using tensor_args_t = GeluBackwardInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GeluBackwardProgramFactory>;
    // validate, compute_output_specs, create_output_tensors, compute_program_hash
};
```

**Invocation** (in `gelu_backward_device_operation.cpp`):
```cpp
namespace ttnn::prim {
Tensor gelu_bw(const Tensor& grad_output, const Tensor& input,
               const std::string& approximate, ...) {
    return ttnn::device_operation::launch<GeluBackwardDeviceOperation>(
        operation_attributes, tensor_args);
}
}
```

#### Circular Buffer Layout

| CB Index | Name | Tiles | Purpose |
|----------|------|-------|---------|
| c_0 | cb_grad_out | 2 | Upstream gradient |
| c_1 | cb_input | 2 | Forward input x |
| c_2 | cb_grad_in | 2 | Output: grad × GELU'(x) |

No intermediate CBs needed. Double-buffered (2 tiles each).

#### Reader/Writer Kernels (Shared)

- **Reader:** `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Writer:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

#### Compute Kernel (`eltwise_bw_gelu_poly.cpp`)

```cpp
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    tile_regs_acquire();
    copy_tile(cb_grad_out, i, 0);    // dest[0] = grad_out
    copy_tile(cb_input, i, 1);       // dest[1] = input
    gelu_derivative_tile<false>(1);  // dest[1] = GELU'(input)
    mul_binary_tile(0, 1, 0);        // dest[0] = grad_out * GELU'(input)
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_grad_in);
    tile_regs_release();
}
```

#### SFPU Math: 5 Piecewise Regions

The `calculate_gelu_derivative_simple()` function in `ckernel_sfpu_gelu.h`:

**Region 1: x ≤ -13.375** — Saturate to 0 (BF16 natural saturation)

**Region 2: (-13.375, -5]** — Fused x·exp(t) with Mills ratio correction:
```cpp
sfpi::vFloat x2 = x * x;
sfpi::vFloat t = x2 * (-0.5f);
sfpi::vFloat x_exp = x_times_exp_negative_tail(x, t);
sfpi::vFloat inv_x2 = _sfpu_reciprocal_<2>(x2);
sfpi::vFloat inv_x4 = inv_x2 * inv_x2;
sfpi::vFloat correction = 1.0f - inv_x2 + inv_x4;
result = x_exp * INV_SQRT_2PI * correction;
```

**Region 3: [-5, -3)** — Degree-8 shifted polynomial (t = x + 4 maps to [-1, 1]):
```cpp
sfpi::vFloat t = x + 4.0f;
result = PolynomialEvaluator::eval(t, LEFT_C0, ..., LEFT_C8);
```

**Region 4: [-3, 3.1719)** — Degree-16 Sollya polynomial:
```cpp
result = PolynomialEvaluator::eval(x, CORE_C0, ..., CORE_C16);
```

**Region 5: x ≥ 3.1719** — Saturate to 1

#### Fused x·exp(t) — Key Innovation

The `x_times_exp_negative_tail()` function avoids intermediate underflow:
```cpp
sfpi_inline sfpi::vFloat x_times_exp_negative_tail(sfpi::vFloat x, sfpi::vFloat t) {
    // Cody-Waite range reduction: t = k·ln(2) + r
    sfpi::vFloat z = t * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);
    sfpi::vFloat r = k * LN2_HI + t;
    r = k * LN2_LO + r;

    // Degree-5 Taylor for exp(r)
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, C2, C3, C4, C5);

    // KEY: Multiply by x BEFORE exponent shift to avoid underflow
    sfpi::vFloat x_poly = x * poly;

    // Exponent bit manipulation
    sfpi::vInt xpoly_exp = sfpi::exexp_nodebias(x_poly);
    sfpi::vInt new_exp = xpoly_exp + k_int;

    // FTZ check on FINAL result
    sfpi::vFloat result = sfpi::vConst0;
    v_if(new_exp > 0) { result = sfpi::setexp(x_poly, new_exp); }
    v_endif;
    return result;
}
```

The insight: `exp(-88.6) ≈ 3e-39` underflows in BF16, but `-13.3 × exp(-88.6) ≈ -4e-38` does NOT. By computing `(x × poly) × 2^k` instead of `x × (poly × 2^k)`, the intermediate stays representable.

#### ULP Results

| Region | Count | Max ULP (Before) | Max ULP (After) |
|--------|-------|----------------:|----------------:|
| x ≤ -13.375 | 15,681 | 32,460 | 0 |
| (-13.375, -5] | 181 | ~6,256 | 1 |
| [-5, -3) | 96 | ~1,886 | 1 |
| [-3, 3.17) | 32,655 | ~146 | 1 |
| x ≥ 3.17 | 15,947 | ~16,203 | 0 |
| **OVERALL** | **64,560** | **32,460** | **1** |

---

### PR #39101: GELU Forward (Max ULP 47 → 2)

**PR:** [#39101](https://github.com/tenstorrent/tt-metal/pull/39101) — `#35290: Fix forward GELU ULP precision with piecewise CDF polynomials`
**State:** OPEN, CHANGES_REQUESTED (perf benchmark pending)
**Branch:** `origin/ivoitovych/issue-35290-gelu-forward-ulp-fix`
**Changes:** +2240 / -116, 7 files

#### Files Changed

| File | Change |
|------|--------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Replaced `calculate_gelu_chebyshev()` with `calculate_gelu_piecewise()` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Same (byte-identical pair) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h` | Added `DST_ACCUM_MODE` passthrough for RNE rounding guard |
| `tests/ttnn/unit_tests/gtests/test_gelu_fw_ulp.cpp` | **NEW** — 11 C++ ULP tests |
| `tests/ttnn/unit_tests/gtests/sources.cmake` | Registered test file |
| `tests/ttnn/unit_tests/operations/eltwise/generate_gelu_fw_coefficients.py` | **NEW** — coefficient generation |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_fw_saturation_research.py` | **NEW** — saturation research |

#### What Was Replaced

The old kernel used a single degree-15 Chebyshev polynomial covering x ∈ [-5.5, +∞), with Max ULP = 32,767 for x < -5.5 (returned 0) and a hardcoded floor of 2.98e-05 near zero.

#### SFPU Math: 5 Piecewise Regions (`calculate_gelu_piecewise()`)

```cpp
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_gelu_piecewise(sfpi::vFloat x) {
    sfpi::vFloat result = sfpi::vConst0;  // Default: 0 for x ≤ -13.1875

    // Region 5: Identity saturation x ≥ 2.78125
    v_if(x >= 2.78125f) { result = x; }

    // Region 4: Core CDF [-3, 2.78125) — degree-13 odd-coefficient polynomial
    v_elseif(x >= -3.0f) {
        sfpi::vFloat u = x * x;  // u=x² factoring: only odd powers needed
        sfpi::vFloat odd_poly = PolynomialEvaluator::eval(
            u,
            GELU_CDF_CORE_C1, GELU_CDF_CORE_C3, GELU_CDF_CORE_C5,
            GELU_CDF_CORE_C7, GELU_CDF_CORE_C9, GELU_CDF_CORE_C11,
            GELU_CDF_CORE_C13);
        sfpi::vFloat phi = GELU_CDF_CORE_C0 + x * odd_poly;  // C0 = 0.5
        result = x * phi;
    }

    // Region 3: Left CDF [-5, -3) — degree-8 shifted polynomial (t = x+4)
    v_elseif(x >= -5.0f) {
        sfpi::vFloat t = x + 4.0f;
        sfpi::vFloat phi = PolynomialEvaluator::eval(
            t, GELU_CDF_LEFT_C0, ..., GELU_CDF_LEFT_C8);
        result = x * phi;
    }

    // Region 2: Exp-based (-13.1875, -5) — Mills ratio asymptotic
    v_elseif(x > -13.1875f) {
        constexpr float K = -0.3989422804014327f;   // -1/sqrt(2π)
        constexpr float K3 = 3.0f * K;
        sfpi::vFloat x2 = x * x;
        sfpi::vFloat t = x2 * (-0.5f);
        sfpi::vFloat exp_val = _sfpu_exp_f32_accurate_(t);
        sfpi::vFloat inv_x2 = _sfpu_reciprocal_<2>(x2);
        sfpi::vFloat scaled = (K3 * inv_x2 - K) * inv_x2 + K;  // 2 MADs + 1 MUL
        result = exp_val * scaled;
    }
    // Region 1: x ≤ -13.1875 saturates to 0
    v_endif;
    return result;
}
```

**Key technique — u=x² factoring:** The CDF Φ(x) = 0.5 + odd_function(x), so only odd-power coefficients are needed. Evaluating via u=x² saves 47% SFPU ops vs the original degree-15 polynomial.

**Core CDF coefficients (degree-13, odd powers + constant):**
```
C0  = 5.000000000e-01
C1  = 3.989379361e-01
C3  = -6.644114224e-02
C5  = 9.881129978e-03
C7  = -1.120736963e-03
C9  = 9.164031378e-05
C11 = -4.721944427e-06
C13 = 1.119074048e-07
```

**RNE rounding guard** (when `is_fp32_dest_acc_en` is false):
```cpp
if constexpr (!is_fp32_dest_acc_en) {
    result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
}
```
Rounds fp32 result to BF16 precision before writing to `dst_reg`, ensuring the packer's rounding matches expectations.

**Infrastructure change** — `DST_ACCUM_MODE` passthrough:
```cpp
// Old: SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_gelu, RC, fast_and_approx, idst)
// New: SFPU_TWO_PARAM_KERNEL(calculate_gelu, fast_and_approx, DST_ACCUM_MODE, idst, RC)
```

**Init function change:**
```cpp
// Old: _init_gelu_<APPROXIMATION_MODE>()
// New: _init_reciprocal_<false, false>()  // accurate mode needs 1/x² for exp region
```

#### ULP Results

| Region | Count | Mean ULP | Max ULP | Worst x |
|--------|-------|----------|---------|---------|
| Saturation to 0 (x ≤ -13.1875) | 15,917 | 0.00 | 0 | — |
| Exp-based (-13.1875, -5] | 179 | 0.03 | 2 | -5.0 |
| Left CDF poly [-5, -3) | 95 | 0.12 | 1 | -4.25 |
| Core CDF poly [-3, 2.78125) | 32,629 | 0.00 | 1 | 2.342e-38 |
| Identity (x ≥ 2.78125) | 16,206 | 0.00 | 0 | — |
| **OVERALL** | **65,026** | **0.00** | **2** | **-5.0** |

Only 1 value out of 65,026 has ULP = 2.

---

### PR #39303: Main API GELU Backward (Max ULP 32,460 → 1)

**PR:** [#39303](https://github.com/tenstorrent/tt-metal/pull/39303) — `#38643: Replace composite gelu_bw(approximate=none) with fused polynomial kernel`
**State:** OPEN, REVIEW_REQUIRED
**Branch:** `origin/ivoitovych/issue-38643-gelu-bw-composite-fused-kernel-pr`
**Changes:** +3207 / -18, 14 files
**This is the most directly applicable blueprint for the tanh_bw fix.**

#### Files Changed

**New files (8):**

| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/gelu_bw_device_operation_types.hpp` | `GeluBwParams` + `GeluBwInputs` |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/gelu_bw_device_operation.hpp` | `GeluBwDeviceOperation` + `launch_gelu_bw()` |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/gelu_bw_device_operation.cpp` | validate, compute_output_specs, create_output_tensors, hash |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/gelu_bw_program_factory.hpp` | Program factory with `shared_variables_t` |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/gelu_bw_program_factory.cpp` | CB setup, kernel creation, work distribution, runtime args |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/device/kernels/compute/eltwise_bw_gelu_poly.cpp` | Compute kernel |
| `tests/ttnn/unit_tests/gtests/test_gelu_bw_main_ulp.cpp` | C++ tests (19 tests) |
| `tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_main_ulp.py` | Python tests (33 tests) |

**Modified files (6):**

| File | Change |
|------|--------|
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` | Replace composite `else` branch with `launch_gelu_bw()` |
| `ttnn/cpp/ttnn/operations/eltwise/unary_backward/CMakeLists.txt` | Add kernel file set + 2 source files |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Add derivative functions |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | Same (byte-identical pair) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h` | Add `gelu_derivative_tile_init<>()` + `gelu_derivative_tile<>()` |
| `tests/ttnn/unit_tests/gtests/sources.cmake` | Register test file |

#### Architecture: 5-Layer Device Operation Pattern

```
Layer 1: Types         → gelu_bw_device_operation_types.hpp
Layer 2: Operation     → gelu_bw_device_operation.hpp/.cpp
Layer 3: Program Fact. → gelu_bw_program_factory.hpp/.cpp
Layer 4: Compute Kern. → kernels/compute/eltwise_bw_gelu_poly.cpp
Layer 5: SFPU API      → gelu.h (tile-level init/compute)
```

**Types** (`gelu_bw_device_operation_types.hpp`):
```cpp
struct GeluBwParams {
    const DataType output_dtype = DataType::INVALID;
    const MemoryConfig output_memory_config;
};

struct GeluBwInputs {
    const Tensor& grad_output;
    const Tensor& input;
    std::optional<Tensor> preallocated_input_grad;
};
```

Note: No `approximate` field — only `approximate="none"` uses the fused kernel; `approximate="tanh"` stays composite.

**Device operation** (`gelu_bw_device_operation.hpp`):
```cpp
struct GeluBwDeviceOperation {
    using operation_attributes_t = GeluBwParams;
    using tensor_args_t = GeluBwInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GeluBwProgramFactory>;
};

// Entry point — callable from unary_backward.cpp
Tensor launch_gelu_bw(const Tensor& grad_output, const Tensor& input,
    DataType output_dtype, const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output);
```

**SFPU API** (`gelu.h`):
```cpp
template <bool fast_and_approx = false>
ALWI void gelu_derivative_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL(gelu_derivative, sfpu::gelu_derivative_polynomial_init,
        fast_and_approx));
}

template <bool fast_and_approx = false>
ALWI void gelu_derivative_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_gelu_derivative_polynomial,
        RC, fast_and_approx, idst));
}
```

#### Circular Buffer Layout

| CB Index | Name | Tiles | Purpose |
|----------|------|-------|---------|
| c_0 | cb_grad_out | 2 | Upstream gradient |
| c_1 | cb_input | 2 | Forward input x |
| c_2 | cb_grad_in | 2 | Output: grad × GELU'(x) |

Same as experimental version. 2-tile double-buffering.

#### Compute Kernel (`eltwise_bw_gelu_poly.cpp`)

```cpp
void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    unary_op_init_common(cb_grad_out, cb_grad_in);
    gelu_derivative_tile_init<false>();
    mul_binary_tile_init();

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_reserve_back(cb_grad_in, per_core_block_size);
        cb_wait_front(cb_grad_out, per_core_block_size);
        cb_wait_front(cb_input, per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            tile_regs_acquire();
            copy_tile(cb_grad_out, i, 0);    // dest[0] = grad_out
            copy_tile(cb_input, i, 1);       // dest[1] = input
            gelu_derivative_tile<false>(1);  // dest[1] = GELU'(input)
            mul_binary_tile(0, 1, 0);        // dest[0] = grad_out * GELU'(input)
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_grad_in);
            tile_regs_release();
        }

        cb_pop_front(cb_grad_out, per_core_block_size);
        cb_pop_front(cb_input, per_core_block_size);
        cb_push_back(cb_grad_in, per_core_block_size);
    }
}
```

#### How `unary_backward.cpp` Was Modified

**BEFORE (composite, 17 lines):**
```cpp
    } else {
        float kAlpha = M_SQRT1_2;
        float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
        Tensor cdf = ttnn::multiply(
            (ttnn::add(
                ttnn::erf(ttnn::multiply(input, kAlpha, std::nullopt, output_memory_config)),
                1, std::nullopt, output_memory_config)),
            0.5);
        Tensor pdf = ttnn::multiply(
            ttnn::exp(ttnn::multiply(ttnn::multiply(input, input), -0.5), false, output_memory_config),
            kBeta, std::nullopt, output_memory_config);
        ttnn::multiply(
            grad, ttnn::add(cdf, ttnn::multiply(input, pdf)), std::nullopt, output_memory_config, input_grad);
        result.push_back(input_grad);
    }
```

**AFTER (fused kernel, 5 lines):**
```cpp
    } else {
        DataType output_dtype = input.dtype();
        auto result_tensor = ttnn::operations::unary_backward::gelu_bw::launch_gelu_bw(
            grad, input, output_dtype, output_memory_config, input_grad);
        result.push_back(result_tensor);
    }
```

Plus include at top: `#include "gelu_bw/device/gelu_bw_device_operation.hpp"`

#### CMakeLists.txt Changes

```cmake
# Added before target_sources:
file(GLOB_RECURSE kernels gelu_bw/device/kernels/*)

# Added to target_sources PUBLIC section:
    FILE_SET kernels TYPE HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${kernels}

# Added to target_sources PRIVATE section:
    gelu_bw/device/gelu_bw_device_operation.cpp
    gelu_bw/device/gelu_bw_program_factory.cpp
```

#### Key Design Decisions

1. **Shared SFPU math, independent device ops.** The polynomial coefficients and `gelu_derivative_tile<>()` live in infrastructure headers (`ckernel_sfpu_gelu.h`, `gelu.h`). The device operation, program factory, and compute kernel are new and independent from the experimental API.

2. **Only `approximate="none"` changed.** The `approximate="tanh"` path stays composite (no precision issue).

3. **FP32 dest accumulation.** The program factory enables `UnpackToDestFp32` for both input CBs — this ensures polynomial evaluation happens in FP32 precision even when inputs/outputs are BF16.

4. **Per-tile processing.** Cannot batch multiple tiles in dest because `gelu_derivative_tile` uses scratch dest registers during polynomial evaluation.

5. **Interleaved-only.** Validates input is not sharded, is interleaved. Sharded support can be added later.

#### Differences from Experimental API

| Aspect | Main API (PR #39303) | Experimental API (PR #36366) |
|--------|---------------------|------------------------------|
| Namespace | `ttnn::operations::unary_backward::gelu_bw` | `ttnn::prim` |
| Entry point | `launch_gelu_bw()` from `invoke()` | Registered as `ttnn::prim::gelu_bw_polynomial` |
| Location | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw/` | `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/` |
| Independence | Completely independent | Separate codebase |
| SFPU math | **Shared** `gelu_derivative_tile<false>()` | **Same shared SFPU primitive** |

Either can be deleted without affecting the other.

#### ULP Results

| Region | Count | Max ULP (Before) | Max ULP (After) |
|--------|-------|----------------:|----------------:|
| x ≤ -13.375 | 15,681 | 0 | 0 |
| (-13.375, -5] exp-based | 181 | 32,460 | 1 |
| [-5, -3) LEFT poly | 96 | ~9,000 | 1 |
| [-3, 3.17) CORE poly | 32,655 | ~100 | 1 |
| x ≥ 3.17 | 15,947 | 0 | 0 |
| **OVERALL** | **65,026** | **32,460** | **1** |

---

### Comparison of All Approaches

| Operation | PR | Pre-Fix Max ULP | Post-Fix Max ULP | Fix Type | Regions |
|-----------|-----|----------------:|----------------:|----------|---------|
| tanh (forward) | — | — | **1** | Fused unary SFPU: minimax poly + sigmoid | 2 |
| gelu (forward) | #39101 | 47 | **2** | Fused unary SFPU: piecewise CDF poly | 5 |
| gelu_bw (experimental) | #36366 | 32,460 | **1** | Fused binary device op: SFPU derivative poly | 5 |
| gelu_bw (main API) | #39303 | 32,460 | **1** | Fused binary device op: SFPU derivative poly | 5 |
| **tanh_bw (this issue)** | — | **15,140** | **TBD** | **Planned: fused binary device op with sech²(x)** | **TBD** |

**Key takeaways:**
1. All successful fixes replace multi-op composite decompositions with fused SFPU kernels that compute the entire function in a single FP32 pass — eliminating intermediate BF16 rounding and catastrophic cancellation.
2. The tanh_bw fix should follow the **PR #39303 pattern** (main API gelu_bw) exactly — same directory structure under `unary_backward/`, same invoke() modification, same CMakeLists.txt pattern, same 6-file device operation structure.
3. The SFPU math for sech²(x) is simpler than GELU'(x) — likely 3 regions (saturation to 0 for |x| > ~4, core polynomial, identity at 0) vs GELU'(x)'s 5 regions with fused exp.

### Shared SFPU Infrastructure Available for tanh_bw

From the three PRs, these shared primitives are confirmed available and working:

| Primitive | Source | Used By |
|-----------|--------|---------|
| `PolynomialEvaluator::eval()` | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h` | All three PRs |
| `_sfpu_exp_f32_accurate_()` | `ckernel_sfpu_exp.h` | GELU fw (#39101), GELU bw derivative (#36366, #39303) |
| `_sfpu_reciprocal_<N>()` | `ckernel_sfpu_recip.h` | GELU fw (#39101), GELU bw derivative (#36366, #39303) |
| `x_times_exp_negative_tail()` | `ckernel_sfpu_gelu.h` (new) | GELU bw derivative (#36366, #39303) |
| `sfpi::float_to_fp16b()` | SFPI built-in | GELU fw (#39101) for RNE rounding guard |
| `sfpi::exexp_nodebias()` | SFPI built-in | `x_times_exp_negative_tail()` |
| `sfpi::setexp()` | SFPI built-in | `x_times_exp_negative_tail()` |
| `v_if/v_elseif/v_endif` | SFPI built-in | All three PRs (piecewise dispatch) |
| `copy_tile()` | Compute API | Backward kernels (load from CB to dest) |
| `mul_binary_tile()` | Compute API | Backward kernels (grad × derivative) |
| `reader_binary_interleaved_start_id.cpp` | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/` | Both backward PRs |
| `writer_unary_interleaved_start_id.cpp` | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/` | Both backward PRs |

### Blueprint: tanh_bw Implementation Plan

Based on the three PR patterns, the tanh_bw fix should create:

**Directory structure** (mirroring PR #39303):
```
ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/
  tanh_bw_device_operation_types.hpp    (TanhBwParams + TanhBwInputs)
  tanh_bw_device_operation.hpp          (TanhBwDeviceOperation + launch_tanh_bw)
  tanh_bw_device_operation.cpp          (validate, output_specs, hash)
  tanh_bw_program_factory.hpp           (TanhBwProgramFactory)
  tanh_bw_program_factory.cpp           (3 CBs, 3 kernels, work distribution)
  kernels/compute/eltwise_bw_tanh_deriv.cpp  (compute kernel)
```

**SFPU math** (in `ckernel_sfpu_tanh.h` or new `ckernel_sfpu_tanh_derivative.h`):
- `calculate_tanh_derivative_polynomial()` computing sech²(x) piecewise
- `tanh_derivative_tile_init<>()` and `tanh_derivative_tile<>()` in compute API header

**Compute kernel pattern** (identical to gelu_bw):
```cpp
copy_tile(cb_grad_out, i, 0);          // dest[0] = grad
copy_tile(cb_input, i, 1);            // dest[1] = input
tanh_derivative_tile<false>(1);       // dest[1] = sech²(input)
mul_binary_tile(0, 1, 0);             // dest[0] = grad × sech²(input)
```

**`unary_backward.cpp` modification** (same pattern as gelu_bw):
```cpp
// Replace lines 289-303 (composite 1-tanh²(x)) with:
DataType output_dtype = input.dtype();
auto result_tensor = ttnn::operations::unary_backward::tanh_bw::launch_tanh_bw(
    grad, input, output_dtype, output_mem_config, input_grad);
```

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
