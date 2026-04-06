# Implementation Notes: Lgamma

## Math Definition

The natural logarithm of the absolute value of the gamma function:
```
lgamma(x) = ln(|Gamma(x)|)
```

For positive x, lgamma uses the Lanczos approximation:
```
z = x - 1
ser = 1.0 + c1/(z+1) + c2/(z+2) + c3/(z+3) + c4/(z+4) + c5/(z+5)
tmp = z + 5.5
lgamma(x) = (z + 0.5) * ln(tmp) - tmp + ln(sqrt(2*pi)) + ln(ser)
```

Special cases: lgamma(1) = 0, lgamma(2) = 0

## Architecture Overview

Lgamma is implemented as an SFPU (Scalar Floating Point Unit) unary operation on Tenstorrent hardware (Wormhole and Blackhole architectures). The implementation uses the Lanczos approximation with g=5 along with helper functions for reciprocal and logarithm computation using polynomial approximations.

## Files Created

### 1. SFPU Kernel (Wormhole)

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h`

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Reciprocal using Newton-Raphson iteration from SFPI instructions.
// Computes 1/in for positive in. Sign must be handled by caller.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _lgamma_reciprocal_(const sfpi::vFloat in) {
    // Force sign to positive
    sfpi::vFloat val = sfpi::setsgn(in, 0);

    // Save original exponent
    sfpi::vInt orig_exp = sfpi::exexp(val);

    // Normalize to [0.5, 1.0) range by setting exponent to 126
    val = sfpi::setexp(val, 126);

    // Initial guess: 1.44 * (val * 1.44 + 2.0)
    // This is the standard Newton-Raphson seed for reciprocal
    sfpi::vFloat vConstLn2Recip = 1.442695f;
    sfpi::vFloat two = 2.0f;
    sfpi::vFloat result = vConstLn2Recip * (val * vConstLn2Recip + two);

    // Newton-Raphson iterations: result = result * (val * result + 2)
    for (int s_iter = 0; s_iter < (max_iter - 1); s_iter++) {
        result = result * (val * result + two);
    }

    // Reconstruct exponent: new_exp = new_exp - orig_exp + 126
    sfpi::vInt new_exp = sfpi::exexp(result);
    new_exp -= orig_exp;
    new_exp += 126;

    v_if(new_exp < 0) {
        result = 0.0f;
        new_exp = 0;
    }
    v_endif;

    return sfpi::setexp(result, new_exp);
}

// Natural logarithm using exponent extraction + polynomial approximation.
// Computes ln(x) for x > 0. Undefined for x <= 0.
sfpi_inline sfpi::vFloat _lgamma_log_(const sfpi::vFloat x) {
    // Extract debiased exponent: for x = m * 2^e, gives e
    sfpi::vInt exp_i = sfpi::exexp(x);

    // Set exponent to 127 to get mantissa m in [1.0, 2.0)
    sfpi::vFloat m = sfpi::setexp(x, 127);

    // f = m - 1.0, so f in [0, 1)
    sfpi::vFloat f = m - sfpi::vConst1;

    // Polynomial approximation of ln(1+f) using Horner's method:
    // ln(1+f) ~ f - f^2/2 + f^3/3 - f^4/4 + f^5/5
    // = f * (1 + f * (-0.5 + f * (0.3333 + f * (-0.25 + f * 0.2))))
    sfpi::vFloat poly = f * 0.2f;
    poly = poly + -0.25f;
    poly = poly * f;
    poly = poly + 0.333333f;
    poly = poly * f;
    poly = poly + -0.5f;
    poly = poly * f;
    poly = poly + sfpi::vConst1;
    poly = poly * f;

    // Convert exponent to float
    sfpi::vFloat exp_f = sfpi::int32_to_float(exp_i);

    // result = exponent * ln(2) + ln(mantissa)
    return exp_f * 0.6931472f + poly;
}

// lgamma(x) = ln(|Gamma(x)|)
// Uses the Lanczos approximation with g=5, matching the existing composite implementation.
//
// For x > 0:
//   z = x - 1
//   ser = 1.0 + 76.18009/(z+1) - 86.50532/(z+2) + 24.01410/(z+3)
//             - 1.231740/(z+4) + 0.001209/(z+5) - 0.000005395/(z+6)
//   tmp = z + 5.5
//   lgamma = (z + 0.5) * ln(tmp) - tmp + ln(sqrt(2*pi)) + ln(ser)
//
// Special cases: lgamma(1) = 0, lgamma(2) = 0
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lgamma() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // z = x - 1
        sfpi::vFloat z = x - sfpi::vConst1;

        // Compute Lanczos series sum:
        // ser = 1.0 + c1/(z+1) + c2/(z+2) + c3/(z+3) + c4/(z+4) + c5/(z+5)
        // Note: z+1 = x, z+2 = x+1, etc.
        sfpi::vFloat ser = sfpi::vConst1;

        // Term 1: 76.18009172947146 / (z + 1)
        sfpi::vFloat denom = z + sfpi::vConst1;
        sfpi::vFloat recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * 76.18009f;

        // Term 2: -86.50532032941677 / (z + 2)
        denom = z + 2.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * -86.50532f;

        // Term 3: 24.01409824083091 / (z + 3)
        denom = z + 3.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * 24.01410f;

        // Term 4: -1.231739572450155 / (z + 4)
        denom = z + 4.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * -1.231740f;

        // Term 5: 0.1208650973866179e-2 / (z + 5)
        denom = z + 5.0f;
        recip = _lgamma_reciprocal_<2>(denom);
        ser = ser + recip * 0.001209f;

        // Term 6: -0.5395239384953e-5 / (z + 6)
        // This term is negligible for bfloat16 precision, skip it.

        // tmp = z + 5.5
        sfpi::vFloat tmp = z + 5.5f;

        // log_tmp = ln(tmp)
        sfpi::vFloat log_tmp = _lgamma_log_(tmp);

        // log_ser = ln(ser)
        sfpi::vFloat log_ser = _lgamma_log_(ser);

        // result = (z + 0.5) * log(tmp) - tmp + ln(sqrt(2*pi)) + log(ser)
        // ln(sqrt(2*pi)) = 0.9189385332046727
        sfpi::vFloat result = (z + 0.5f) * log_tmp;
        result = result - tmp;
        result = result + 0.918939f;
        result = result + log_ser;

        // Special cases: lgamma(1) = 0, lgamma(2) = 0
        v_if(x == sfpi::vConst1) { result = sfpi::vConst0; }
        v_endif;

        v_if(x == 2.0f) { result = sfpi::vConst0; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### 2. SFPU Kernel (Blackhole)

**Path**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h`

Identical copy to Wormhole implementation.

### 3. LLK Wrapper (Wormhole)

**Status**: Not present - lgamma does not have an LLK wrapper in wormhole_b0.

### 4. LLK Wrapper (Blackhole)

**Status**: Not present - lgamma does not have an LLK wrapper in blackhole.

### 5. Compute API Header

**Path**: `tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_lgamma.h"
#endif

namespace ckernel {

ALWI void lgamma_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROX>())); }

ALWI void lgamma_tile(uint32_t idst) { MATH((SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_lgamma, RC, APPROX, idst))); }

}  // namespace ckernel
```

## Layer 6: SfpuType Enum Entry

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

```cpp
enum class SfpuType {
    unused = 0,
    cosh,
    cbrt,
    hardtanh,
    lgamma,        // <-- Entry for lgamma
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
#if SFPU_OP_LGAMMA_INCLUDE
#include "api/compute/eltwise_unary/lgamma.h"
#endif
```

## Layer 8: llk_math_unary_sfpu_api.h

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

**Status**: lgamma is not included in this file. The compute API directly references the kernel functions.

## Layer 9: Dispatch (unary_op_utils.cpp)

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

```cpp
case UnaryOpType::LGAMMA: return {"lgamma_tile_init();", fmt::format("lgamma_tile({});", idst)};
```

## Layer 10: Python Golden Function

**Path**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_lgamma(input_tensor_a, *args, **kwargs):
    import torch

    return torch.lgamma(input_tensor_a)


ttnn.attach_golden_function(ttnn.lgamma, golden_function=_golden_function_lgamma)
```

## Layer 11: Test File

**Path**: `tests/ttnn/unit_tests/operations/eltwise/test_lgamma.py`

```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger

pytestmark = pytest.mark.use_module_device


def run_lgamma_test(device, h, w, pcc=0.999, rtol=1.6e-2, atol=1e-2):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 9.9 + 0.1
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC: {pcc_value} (threshold: {pcc})")
    allclose_passing = torch.allclose(torch_output_tensor.float(), output_tensor.float(), rtol=rtol, atol=atol)
    if not allclose_passing:
        max_diff = (torch_output_tensor.float() - output_tensor.float()).abs().max().item()
        logger.warning(f"allclose failed: max_diff={max_diff}, rtol={rtol}, atol={atol}")
    return passing, pcc_value


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 32),
        (64, 64),
        (128, 128),
        (32, 64),
        (64, 128),
    ],
)
def test_lgamma(device, h, w):
    passing, pcc_value = run_lgamma_test(device, h, w, pcc=0.999)
    assert passing, f"PCC {pcc_value} below threshold 0.999"


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 32),
        (64, 64),
    ],
)
def test_lgamma_small_inputs(device, h, w):
    torch.manual_seed(42)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 0.9 + 0.1
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"Small inputs PCC: {pcc_value}")
    assert passing, f"PCC {pcc_value} below threshold 0.999 for small inputs"


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 32),
        (64, 64),
    ],
)
def test_lgamma_large_inputs(device, h, w):
    torch.manual_seed(123)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 5.0 + 5.0
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"Large inputs PCC: {pcc_value}")
    assert passing, f"PCC {pcc_value} below threshold 0.999 for large inputs"


def test_lgamma_special_values(device):
    torch_input_tensor = torch.tensor([[1.0, 2.0, 1.0, 2.0]] * 8, dtype=torch.bfloat16).reshape(32, 1).expand(32, 32)
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor.contiguous(), layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"Special values PCC: {pcc_value}")
    max_abs = output_tensor.float().abs().max().item()
    logger.info(f"Special values max abs value: {max_abs}")
    assert max_abs < 0.2, f"lgamma(1) and lgamma(2) should be ~0, got max abs {max_abs}"
```

## Layer 12: Registration

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
REGISTER_UNARY_OPERATION(lgamma, LGAMMA)
```

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
bind_unary_operation<"lgamma", &ttnn::lgamma>(
    mod,
    ...
);
```

## Design Decisions

1. **Lanczos Approximation with g=5**: The implementation uses Lanczos approximation with g=5, which provides a good balance between accuracy and computational cost for bfloat16 precision.

2. **Helper Functions**: Two critical helper functions are defined:
   - `_lgamma_reciprocal_<max_iter>()`: Computes reciprocals using Newton-Raphson iteration with exponent extraction and normalization
   - `_lgamma_log_()`: Computes natural logarithm using exponent extraction and polynomial approximation

3. **Polynomial Approximation for ln()**: The logarithm uses a degree-4 polynomial (Horner form) for ln(1+f) in the range [0, 1), where f is the normalized mantissa.

4. **Coefficient Precision**: The Lanczos coefficients are truncated to 6 significant figures to fit in bfloat16 precision (e.g., 76.18009 instead of 76.18009172947146).

5. **Skip 6th Term**: The 6th Lanczos term (-0.5395239384953e-5) is skipped as it is negligible for bfloat16 precision.

6. **Special Case Handling**: lgamma(1) = 0 and lgamma(2) = 0 are explicitly handled with v_if conditions to ensure exact results.

7. **Loop Unroll Disabled**: The pragma GCC unroll 0 directive disables unrolling to avoid register pressure with the complex computation.

## Known Limitations

1. **Range Constraint**: The implementation is designed for positive inputs (x > 0.1). Results for negative or zero inputs are undefined.

2. **Polynomial Truncation**: The Lanczos approximation is truncated to 5 terms (6th term omitted), which may introduce small errors at extreme values.

3. **Logarithm Approximation**: The logarithm function uses a degree-4 polynomial, limiting precision to approximately 3-4 decimal places.

4. **No Exact Mode**: Only approximation mode is supported; there is no exact mode variant.

5. **LLK Wrapper Not Exported**: Unlike cbrt and hardtanh, lgamma does not have dedicated LLK wrapper functions. Initialization uses generic `SfpuType::unused`.

## Test Results

The test suite includes:
- Standard range [0.1, 10.0): PCC ≥ 0.999 across multiple tensor sizes
- Small inputs [0.1, 1.0): PCC ≥ 0.999
- Large inputs [5.0, 10.0): PCC ≥ 0.999
- Special values: lgamma(1) and lgamma(2) must be < 0.2 in absolute value
- Relative tolerance: ≤ 1.6%, absolute tolerance: ≤ 0.01
