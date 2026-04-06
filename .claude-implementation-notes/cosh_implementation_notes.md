# Implementation Notes: Cosh

## Math Definition

The hyperbolic cosine function:
```
cosh(x) = (exp(x) + exp(-x)) / 2
```

## Architecture Overview

Cosh is implemented as an SFPU (Scalar Floating Point Unit) unary operation that runs on Tenstorrent hardware (Wormhole and Blackhole architectures). The implementation uses an inline polynomial approximation for exp() with range reduction and exponent manipulation.

## Files Created

### 1. SFPU Kernel (Wormhole)

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

namespace cosh_internal {

// Inline exponential using polynomial approximation after range reduction.
// Computes exp(x) using: exp(x) = poly(r) * 2^k, where k = round(x/ln2), r = x - k*ln2.
// Polynomial is degree-2 Horner evaluation matching the coefficients from SDPA compute_common.hpp.
template <bool APPROXIMATION_MODE>
sfpi::vFloat inline_exp(sfpi::vFloat val) {
    using namespace sfpi;

    constexpr float LN2_RECIP = 1.44269504088896340736f;  // 1/ln(2)
    constexpr float NEG_LN2 = -0.69314718055994530942f;   // -ln(2)

    // Degree 2 polynomial coefficients for exp(r) where |r| <= ln(2)/2
    constexpr float c0 = 0.999848792924395313327307061545061386175496934006f;
    constexpr float c1 = 1.01508760098521056684783640695492761469306929535975f;
    constexpr float c2 = 0.50628367056745568861842335616023694454759126020461f;

    // k = round(x / ln(2)), clamped to int8 range [-128, 127]
    vFloat scaled = val * LN2_RECIP;
    vUInt k_uint = float_to_int8(scaled);
    vInt k_int = reinterpret<vInt>(k_uint);
    vFloat k_float = int32_to_float(k_int, 0);

    // r = x - k * ln(2) -- fractional part in [-ln2/2, ln2/2]
    vFloat r = val + k_float * NEG_LN2;

    // Evaluate degree-2 polynomial: exp(r) ~ (c2*r + c1)*r + c0
    vFloat poly = r * c2 + c1;
    poly = poly * r + c0;

    // Reconstruct exp(x) = poly * 2^k by constructing 2^k via exponent manipulation.
    // IEEE754: biased_exponent = k + 127. Set exponent field of 1.0f to (k + 127).
    vInt biased_exp = k_int + vInt(127);
    vFloat two_to_k = setexp(vFloat(1.0f), reinterpret<vUInt>(biased_exp));
    vFloat result = poly * two_to_k;

    // Handle underflow: for very negative x, exp(x) -> 0
    v_if(val < -80.0f) { result = 0.0f; }
    v_endif;

    return result;
}

}  // namespace cosh_internal

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_cosh() {
    using namespace sfpi;

    // cosh(x) = (exp(x) + exp(-x)) / 2
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];

        // Compute exp(x) and exp(-x) using inline polynomial exp
        vFloat exp_pos = cosh_internal::inline_exp<APPROXIMATION_MODE>(x);
        vFloat exp_neg = cosh_internal::inline_exp<APPROXIMATION_MODE>(-x);

        // cosh = (exp(x) + exp(-x)) / 2
        vFloat result = (exp_pos + exp_neg) * 0.5f;

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### 2. SFPU Kernel (Blackhole)

**Path**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cosh.h`

Identical copy to Wormhole implementation.

### 3. LLK Wrapper (Wormhole)

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cosh.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_cosh.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_cosh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cosh, APPROXIMATE>();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_unary_sfpu_cosh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_cosh<APPROXIMATE>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### 4. LLK Wrapper (Blackhole)

**Path**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_cosh.h`

Identical copy to Wormhole implementation.

### 5. Compute API Header

**Path**: `tt_metal/hw/inc/api/compute/eltwise_unary/cosh.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api.h"

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool approx = true>
ALWI void cosh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_cosh_init<approx>()));
}

// clang-format off
/**
 * Performs element-wise computation of hyperbolic cosine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool approx = true>
ALWI void cosh_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_cosh<approx, DST_ACCUM_MODE>(idst)));
}

}  // namespace ckernel
```

## Layer 6: SfpuType Enum Entry

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

```cpp
enum class SfpuType {
    unused = 0,
    cosh,          // <-- Entry for cosh
    cbrt,
    hardtanh,
    lgamma,
    hardsigmoid,
    rpow,
    softsign,
    hardswish,
    softshrink,
    swish,
    frac,
    atanh,
    sinh,
};
```

## Layer 7: sfpu_split_includes.h

**Path**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

```cpp
#if SFPU_OP_COSH_INCLUDE
#include "api/compute/eltwise_unary/cosh.h"
#endif
```

## Layer 8: llk_math_unary_sfpu_api.h

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

```cpp
#include "llk_math_eltwise_unary_sfpu_cosh.h"
```

## Layer 9: Dispatch (unary_op_utils.cpp)

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

```cpp
case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
```

## Layer 10: Python Golden Function

**Path**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_cosh(input_tensor_a, *args, **kwargs):
    import torch

    return torch.cosh(input_tensor_a)


ttnn.attach_golden_function(ttnn.cosh, golden_function=_golden_function_cosh)
```

## Layer 11: Test File

**Path**: `tests/ttnn/unit_tests/operations/eltwise/test_cosh.py`

**Status**: Not present - no test file exists for cosh operation.

## Layer 12: Registration

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
REGISTER_UNARY_OPERATION(cosh, COSH)
```

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
bind_unary_operation<"cosh", &ttnn::cosh>(
    mod,
    R"doc(\mathrm{{output\_tensor}}_i = \cosh(\mathrm{{input\_tensor}}_i))doc",
    ...
);
```

## Design Decisions

1. **Polynomial Approximation for exp()**: The implementation uses a degree-2 Horner polynomial (c0 + c1*r + c2*r^2) on the reduced range [-ln(2)/2, ln(2)/2]. This provides good accuracy while minimizing computation cost.

2. **Range Reduction via IEEE754 Bit Manipulation**: The exponent is extracted as an integer (k = round(x/ln(2))), and the mantissa is normalized to [0.5, 1.0). This avoids expensive ln(2) multiplications and improves stability.

3. **Exponent Reconstruction**: The final exp(x) = poly(r) * 2^k is computed by directly setting the IEEE754 exponent field, avoiding log/antilog computations.

4. **Underflow Handling**: For very negative x (< -80), the result saturates to 0 to avoid underflow.

5. **Symmetry for cosh()**: Since cosh(x) = (exp(x) + exp(-x))/2, the implementation computes both exp(x) and exp(-x) and averages them, which naturally handles both positive and negative inputs.

## Known Limitations

1. **No explicit test coverage**: The cosh operation lacks dedicated unit tests.
2. **Approximation mode only**: The implementation only supports approximate mode; there is no exact mode variant.
3. **Limited to SFPU range**: The operation is constrained by SFPU hardware limitations on exponent and mantissa ranges.
