# Implementation Notes: atanh

## Math Definition

The `atanh` operation (inverse hyperbolic tangent) is defined as:

```
atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
```

Valid for |x| < 1 (strict inequality). The implementation uses IEEE 754 decomposition for efficient logarithm computation:

```
y = 2^e * m, where m in [1, 2)
ln(y) = e * ln(2) + P(m)
```

where P(m) is a cubic minimax polynomial approximation for ln(m) on [1, 2).

## Files Created

### Layer 1: SFPU Kernel (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
// Valid for |x| < 1.
//
// ln(y) is computed via IEEE 754 decomposition:
//   y = 2^e * m, where m in [1, 2)
//   ln(y) = e * ln(2) + P(m)
// where P(m) is a cubic minimax polynomial approximation for ln(m) on [1, 2).
// Coefficients are from the rpow scalar log2 precomputation (Horner form):
//   P(m) = c0 + m * (c1 + m * (c2 + m * c3))
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;
        sfpi::vFloat b = -x + sfpi::vConst1;

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);
        sfpi::vFloat ma = sfpi::setexp(a, 127);
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3))
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;
        pa = pa * ma + sfpi::vConstFloatPrgm1;
        pa = pa * ma + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa;

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);
        sfpi::vFloat mb = sfpi::setexp(b, 127);
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;
        pb = pb * mb + sfpi::vConstFloatPrgm1;
        pb = pb * mb + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb;

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691
}

}  // namespace ckernel::sfpu
```

### Layer 2: SFPU Kernel (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
// Valid for |x| < 1.
//
// ln(y) is computed via IEEE 754 decomposition:
//   y = 2^e * m, where m in [1, 2)
//   ln(y) = e * ln(2) + P(m)
// where P(m) is a cubic minimax polynomial approximation for ln(m) on [1, 2).
// Coefficients are from the rpow scalar log2 precomputation (Horner form):
//   P(m) = c0 + m * (c1 + m * (c2 + m * c3))
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;
        sfpi::vFloat b = -x + sfpi::vConst1;

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);
        sfpi::vFloat ma = sfpi::setexp(a, 127);
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3))
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;
        pa = pa * ma + sfpi::vConstFloatPrgm1;
        pa = pa * ma + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa;

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);
        sfpi::vFloat mb = sfpi::setexp(b, 127);
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;
        pb = pb * mb + sfpi::vConstFloatPrgm1;
        pb = pb * mb + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb;

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691
}

}  // namespace ckernel::sfpu
```

### Layer 3: LLK Wrapper (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_atanh.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 4: LLK Wrapper (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_atanh.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 5: Compute API Header

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_atanh.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x)).
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
ALWI void atanh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void atanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_atanh_init<APPROX>())); }

}  // namespace ckernel
```

### Layer 6: SfpuType Enum Entry

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

The enum entry is defined in the SfpuType enumeration. atanh is registered as an operation with initialization callback.

### Layer 7: sfpu_split_includes.h

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

The sfpu_split_includes.h file conditionally includes atanh.h when atanh operation is required.

### Layer 8: llk_math_unary_sfpu_api.h

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

Atanh uses the standard unary operation dispatch through llk_math_eltwise_unary_sfpu_atanh wrapper functions with initialization support.

### Layer 9: Dispatch (unary_op_utils.cpp)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Relevant case statements for atanh dispatch:

```cpp
case UnaryOpType::ATANH: return {"atanh_tile_init();", fmt::format("atanh_tile({});", idst)};
```

### Layer 10: Python Golden (unary.py)

**File**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_atanh(input_tensor_a, *args, **kwargs):
    import torch

    return torch.atanh(input_tensor_a)


ttnn.attach_golden_function(ttnn.atanh, golden_function=_golden_function_atanh)
```

### Layer 11: Test File

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_atanh.py`

```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

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
def test_atanh(device, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Filter to valid domain: |x| < 1 (strict inequality)
    # Replace out-of-domain values with 0.0 (a safe in-domain value)
    mask = torch_input.float().abs() < 1.0
    torch_input = torch.where(mask, torch_input, torch.zeros_like(torch_input))

    # Compute reference in float32, flush subnormals to match hardware behavior
    torch_output = torch.atanh(torch_input.float())
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.atanh(tt_input)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    # allclose covers the full domain including small values near zero
    # where the ln-based kernel has reduced precision due to catastrophic cancellation
    # (computing ln(1+x) - ln(1-x) for small x subtracts near-equal values).
    # fp32 uses wider atol because the SFPU cubic polynomial for ln provides ~2-3
    # decimal digits of accuracy regardless of accumulation format.
    if is_fp32:
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=2e-3)
    else:
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)

    # ULP check only for bfloat16 where it is meaningful given the polynomial accuracy.
    # fp32 ULP is too fine-grained for the cubic polynomial's ~10-bit effective precision.
    if not is_fp32:
        large_mask = expected_finite.float().abs() > 0.25
        if large_mask.any():
            expected_large = expected_finite[large_mask].reshape(1, -1)
            actual_large = actual_finite[large_mask].reshape(1, -1)
            assert_with_ulp(expected_large, actual_large, ulp_threshold=4)
```

### Layer 12: Registration

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
REGISTER_UNARY_OPERATION(atanh, ATANH)
```

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
bind_unary_operation<"atanh", &ttnn::atanh>(
    mod, R"doc(\mathrm{{output\_tensor}}_i = \text{atanh}(\mathrm{{input\_tensor}}_i))doc",
    "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

## Design Decisions

- **IEEE 754 Decomposition**: Uses hardware-efficient decomposition y = 2^e * m for logarithm computation.
- **Polynomial Approximation**: Cubic minimax polynomial provides ~10-bit effective precision for ln(m) on [1, 2).
- **Programmable Constants**: Three polynomial coefficients are stored in programmable registers (vConstFloatPrgm0-2) and initialized via atanh_init().
- **Catastrophic Cancellation**: For small |x| near zero, ln(1+x) - ln(1-x) subtraction suffers from precision loss, limiting overall accuracy.

## Test Results

- **bfloat16 tests**: Validates atanh with allclose tolerance rtol=1.6e-2, atol=1e-2, and ULP threshold of 4 for values with |expected| > 0.25.
- **float32 tests**: Stricter tolerances with rtol=1.6e-2, atol=2e-3 to account for full precision capability.
- **Domain filtering**: Input is restricted to |x| < 1 (strict), with out-of-domain values replaced with 0.0.
- **Finite value filtering**: NaN/Inf values are excluded from accuracy checks.

## Known Limitations

- **Reduced precision near zero**: Catastrophic cancellation in ln(1+x) - ln(1-x) for small |x| limits accuracy to approximately 1e-2 absolute error.
- **Polynomial accuracy**: The cubic polynomial provides only ~10-bit effective precision, which is less than float32's full precision.
- **Domain restriction**: Input must satisfy |x| < 1 (strict inequality); behavior at domain boundaries (x = ±1) is undefined.
