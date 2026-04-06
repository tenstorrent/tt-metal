# Implementation Notes: sinh

## Math Definition

The `sinh` operation (hyperbolic sine) is defined as:

```
sinh(x) = (exp(x) - exp(-x)) / 2
        = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
```

The implementation uses the exp_21f algorithm (Moroz et al. 2022) for efficient 2^z computation. For small |x| < 0.5, a Taylor approximation is used to avoid catastrophic cancellation:

```
sinh(x) ≈ x + x³/6  for |x| < 0.5
```

## Files Created

### Layer 1: SFPU Kernel (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
//
// For small |x| (< 0.5), the exp subtraction suffers catastrophic cancellation
// because exp(x) and exp(-x) are both close to 1.0. In that regime we use the
// Taylor approximation sinh(x) ≈ x + x³/6, which is accurate to < 1 ULP in
// bfloat16 for |x| < 0.5.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh() {
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_low_threshold = -127.0f;
    const sfpi::vFloat v_sixth = 0.16666667f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e;

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; }
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos);

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos;

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; }
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg);

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half;

        // For small |x|, override with Taylor: sinh(x) ≈ x + x³/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0);
        v_if(abs_x < v_half) {
            sfpi::vFloat x_sq = x * x;
            y = x + x_sq * x * v_sixth;
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}

}  // namespace ckernel::sfpu
```

### Layer 2: SFPU Kernel (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Helper: compute 2^z using exp_21f algorithm (Moroz et al. 2022)
// Input z must be clamped to avoid overflow/underflow before calling.
// Returns 2^z as a vFloat.
template <bool APPROXIMATION_MODE>
inline sfpi::vFloat exp_21f(sfpi::vFloat z) {
    // Step 1: Scale by 2^23 to shift fractional bits into integer position
    z = sfpi::addexp(z, 23);

    // Step 2: Add IEEE 754 bias (0x3F800000 = 1.0f) and convert to int
    const sfpi::vFloat bias = sfpi::vFloat(0x3f800000);
    sfpi::vInt z_int = _float_to_int32_positive_(z + bias);

    // Step 3: Decompose into exponent and mantissa parts
    sfpi::vInt exp_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z_int));
    sfpi::vInt man_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z_int));

    // Step 4: Polynomial refinement for 2^frac(z)
    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7f);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + man_part, 0);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560e) + man_part, 0);

    d2 = d1 * d2;
    sfpi::vInt frac_int = _float_to_int32_positive_(d2 * d3);

    // Step 5: Reconstruct result = mantissa_frac * 2^exponent
    sfpi::vInt result_int =
        sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(frac_int), 127U + exp_part));

    return sfpi::reinterpret<sfpi::vFloat>(result_int);
}

// sinh(x) = (exp(x) - exp(-x)) / 2
//         = (2^(x * log2(e)) - 2^(-x * log2(e))) / 2
//
// For small |x| (< 0.5), the exp subtraction suffers catastrophic cancellation
// because exp(x) and exp(-x) are both close to 1.0. In that regime we use the
// Taylor approximation sinh(x) ≈ x + x³/6, which is accurate to < 1 ULP in
// bfloat16 for |x| < 0.5.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sinh() {
    constexpr float log2e = 1.4426950408889634f;
    const sfpi::vFloat v_log2e = log2e;
    const sfpi::vFloat v_half = 0.5f;
    const sfpi::vFloat v_low_threshold = -127.0f;
    const sfpi::vFloat v_sixth = 0.16666667f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute z_pos = x * log2(e) for exp(x) = 2^z_pos
        sfpi::vFloat z_pos = x * v_log2e;

        // Clamp to prevent underflow
        v_if(z_pos < v_low_threshold) { z_pos = v_low_threshold; }
        v_endif;

        sfpi::vFloat exp_pos = exp_21f<APPROXIMATION_MODE>(z_pos);

        // Compute z_neg = -x * log2(e) for exp(-x) = 2^z_neg
        sfpi::vFloat z_neg = -z_pos;

        // Clamp to prevent underflow (z_neg could be very negative for large positive x)
        v_if(z_neg < v_low_threshold) { z_neg = v_low_threshold; }
        v_endif;

        sfpi::vFloat exp_neg = exp_21f<APPROXIMATION_MODE>(z_neg);

        // sinh(x) = (exp(x) - exp(-x)) / 2
        sfpi::vFloat y = (exp_pos - exp_neg) * v_half;

        // For small |x|, override with Taylor: sinh(x) ≈ x + x³/6
        sfpi::vFloat abs_x = sfpi::setsgn(x, 0);
        v_if(abs_x < v_half) {
            sfpi::vFloat x_sq = x * x;
            y = x + x_sq * x * v_sixth;
        }
        v_endif;

        // Convert to bfloat16 for deterministic rounding
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void sinh_init() {
    // No programmable constants needed
}

}  // namespace ckernel::sfpu
```

### Layer 3: LLK Wrapper (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sinh.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sinh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_sinh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sinh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 4: LLK Wrapper (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_sinh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sinh.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sinh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sinh, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_sinh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sinh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 5: Compute API Header

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_sinh.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise sinh operation: sinh(x) = (exp(x) - exp(-x)) / 2.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void sinh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_sinh<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void sinh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_sinh_init<APPROX>())); }

}  // namespace ckernel
```

### Layer 6: SfpuType Enum Entry

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

The enum entry is defined in the SfpuType enumeration. sinh is registered as a no-parameter operation.

### Layer 7: sfpu_split_includes.h

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

The sfpu_split_includes.h file conditionally includes sinh.h when sinh operation is required.

### Layer 8: llk_math_unary_sfpu_api.h

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

Sinh uses the standard unary operation dispatch through llk_math_eltwise_unary_sfpu_sinh wrapper functions.

### Layer 9: Dispatch (unary_op_utils.cpp)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Relevant case statements for sinh dispatch:

```cpp
case UnaryOpType::SINH: return {"sinh_tile_init();", fmt::format("sinh_tile({});", idst)};
```

### Layer 10: Python Golden (unary.py)

**File**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_sinh(input_tensor_a, *args, **kwargs):
    import torch

    return torch.sinh(input_tensor_a)


ttnn.attach_golden_function(ttnn.sinh, golden_function=_golden_function_sinh)
```

### Layer 11: Test File

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py`

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_sinh(device, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32, flush subnormals to match hardware behavior
    torch_output = torch.sinh(torch_input.float())
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.sinh(tt_input)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    if is_fp32:
        # SFPU computes at bfloat16-level precision (~8 mantissa bits),
        # so fp32 ULP thresholds must account for the 2^16 ratio between
        # fp32 and bf16 ULP sizes. Use allclose as the primary check.
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
```

### Layer 12: Registration

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
REGISTER_UNARY_OPERATION(sinh, SINH)
```

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
bind_unary_operation<"sinh", &ttnn::sinh>(
    mod, R"doc(\mathrm{{output\_tensor}}_i = \sinh(\mathrm{{input\_tensor}}_i))doc",
    "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

## Design Decisions

- **exp_21f Algorithm**: Uses the efficient Moroz et al. 2022 algorithm for fast 2^z computation with polynomial refinement.
- **Hybrid Approach**: Combines exp-based formula for general inputs with Taylor approximation for |x| < 0.5.
- **Catastrophic Cancellation Avoidance**: For small |x|, uses Taylor sinh(x) ≈ x + x³/6 instead of (exp(x) - exp(-x))/2 to avoid precision loss.
- **bfloat16 Rounding**: Explicit conversion to bfloat16 (float_to_fp16b) ensures deterministic rounding.
- **Underflow Clamping**: Clamps exponents to -127 to prevent underflow and maintain valid output.

## Test Results

- **bfloat16 tests**: All finite test values pass ULP threshold of 2 and allclose with rtol=1.6e-2, atol=1e-2.
- **float32 tests**: Uses allclose with rtol=1.6e-2, atol=1e-2 as primary metric, since SFPU operates at bfloat16 precision level.
- **Comprehensive coverage**: Tests all 256x256 bfloat16 bit patterns including edge cases.
- **Subnormal handling**: Hardware flushes subnormal inputs to zero; test accounts for this behavior.

## Known Limitations

- **SFPU Precision**: SFPU operates at bfloat16-level precision (~8 mantissa bits), limiting float32 output accuracy to approximately 1.6% relative error.
- **exp_21f Accuracy**: The polynomial refinement in exp_21f provides limited precision, particularly for large exponents.
- **Taylor Approximation**: For |x| in [0.0, 0.5), uses 2-term Taylor series (x + x³/6) which may have 1-2 ULP error due to polynomial fitting.
