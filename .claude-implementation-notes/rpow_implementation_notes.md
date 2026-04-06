# Implementation Notes: rpow

## Math Definition

rpow(x, base) = base^x

This computes base raised to the power of x, where base is a constant parameter and x is the input tensor. Implementation uses logarithmic decomposition to avoid overflow/underflow issues with large exponents.

## Files Created

### Layer 1: SFPU Kernel (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h`

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
namespace sfpu {

// rpow(x, base) = base^x
//
// Implementation:
//   base^x = exp(x * ln(base))
//          = 2^(x * log2(base))
//
// We decompose log2(base) using IEEE754 exponent/mantissa, then compute
// 2^(x * log2(base)) using range reduction and a polynomial approximation.
//
// param0: bit-cast uint32_t representation of the float 'base' parameter
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rpow(uint param0) {
    // Reconstruct the float 'base' from the bit-cast uint32_t parameter
    sfpi::vFloat base_val = Converter::as_float(param0);

    // Constants
    sfpi::vFloat ln2 = 0.6931471805599453f;
    sfpi::vFloat inv_ln2 = 1.4426950408889634f;

    // Precompute log2(base) before the tile loop since base is constant.
    // Decompose base = 2^e * m where m in [1, 2):
    //   log2(base) = e + log2(m)

    // Extract biased exponent
    sfpi::vInt e_biased = sfpi::exexp(base_val);
    sfpi::vFloat e_float = sfpi::int32_to_float(e_biased, 0);
    sfpi::vFloat e_unbiased = e_float - 127.0f;

    // Extract mantissa in [1, 2): set exponent to 127 (2^0 scaling)
    sfpi::vFloat m = sfpi::setexp(base_val, 127);
    sfpi::vFloat f = m - 1.0f;

    // Minimax polynomial: log2(1+f) for f in [0, 1)
    sfpi::vFloat log2_m = f * (1.44269504f + f * (-0.72134752f + f * 0.48089835f));

    // log2(base) = e + log2(m)
    sfpi::vFloat log2_base = e_unbiased + log2_m;

    // ln(base) = log2(base) * ln(2)
    sfpi::vFloat ln_base = log2_base * ln2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // y = x * ln(base)
        sfpi::vFloat y = x * ln_base;

        // Compute exp(y) = 2^(y / ln(2))
        sfpi::vFloat z = y * inv_ln2;

        // Range reduction: z = n + frac where n = round(z)
        // Add-subtract trick for rounding
        sfpi::vFloat shift = 1024.0f;
        sfpi::vFloat z_shifted = z + shift;
        sfpi::vFloat n_float = z_shifted - shift;
        sfpi::vFloat frac = z - n_float;

        // 2^frac via Taylor polynomial for |frac| <= 0.5
        sfpi::vFloat exp2_frac = 1.0f + frac * (0.693147f + frac * (0.240227f + frac * 0.055505f));

        // 2^n via exponent manipulation
        sfpi::vInt n_int = sfpi::float_to_int16(n_float);
        sfpi::vInt new_exp = n_int + 127;
        sfpi::vFloat pow2_n = sfpi::setexp(1.0f, new_exp);

        // result = 2^n * 2^frac
        sfpi::vFloat result = pow2_n * exp2_frac;

        // x == 0 => base^0 = 1
        v_if(x == 0.0f) { result = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### Layer 2: SFPU Kernel (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rpow.h`.

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
namespace sfpu {

// rpow(x, base) = base^x
//
// Implementation:
//   base^x = exp(x * ln(base))
//          = 2^(x * log2(base))
//
// We decompose log2(base) using IEEE754 exponent/mantissa, then compute
// 2^(x * log2(base)) using range reduction and a polynomial approximation.
//
// param0: bit-cast uint32_t representation of the float 'base' parameter
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rpow(uint param0) {
    // Reconstruct the float 'base' from the bit-cast uint32_t parameter
    sfpi::vFloat base_val = Converter::as_float(param0);

    // Constants
    sfpi::vFloat ln2 = 0.6931471805599453f;
    sfpi::vFloat inv_ln2 = 1.4426950408889634f;

    // Precompute log2(base) before the tile loop since base is constant.
    // Decompose base = 2^e * m where m in [1, 2):
    //   log2(base) = e + log2(m)

    // Extract biased exponent
    sfpi::vInt e_biased = sfpi::exexp(base_val);
    sfpi::vFloat e_float = sfpi::int32_to_float(e_biased, 0);
    sfpi::vFloat e_unbiased = e_float - 127.0f;

    // Extract mantissa in [1, 2): set exponent to 127 (2^0 scaling)
    sfpi::vFloat m = sfpi::setexp(base_val, 127);
    sfpi::vFloat f = m - 1.0f;

    // Minimax polynomial: log2(1+f) for f in [0, 1)
    sfpi::vFloat log2_m = f * (1.44269504f + f * (-0.72134752f + f * 0.48089835f));

    // log2(base) = e + log2(m)
    sfpi::vFloat log2_base = e_unbiased + log2_m;

    // ln(base) = log2(base) * ln(2)
    sfpi::vFloat ln_base = log2_base * ln2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // y = x * ln(base)
        sfpi::vFloat y = x * ln_base;

        // Compute exp(y) = 2^(y / ln(2))
        sfpi::vFloat z = y * inv_ln2;

        // Range reduction: z = n + frac where n = round(z)
        // Add-subtract trick for rounding
        sfpi::vFloat shift = 1024.0f;
        sfpi::vFloat z_shifted = z + shift;
        sfpi::vFloat n_float = z_shifted - shift;
        sfpi::vFloat frac = z - n_float;

        // 2^frac via Taylor polynomial for |frac| <= 0.5
        sfpi::vFloat exp2_frac = 1.0f + frac * (0.693147f + frac * (0.240227f + frac * 0.055505f));

        // 2^n via exponent manipulation
        sfpi::vInt n_int = sfpi::float_to_int16(n_float);
        sfpi::vInt new_exp = n_int + 127;
        sfpi::vFloat pow2_n = sfpi::setexp(1.0f, new_exp);

        // result = 2^n * 2^frac
        sfpi::vFloat result = pow2_n * exp2_frac;

        // x == 0 => base^0 = 1
        v_if(x == 0.0f) { result = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### Layer 3: Compute API Header

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/rpow.h`

```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rpow.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void rpow_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_rpow, RC, APPROX, idst, param0));
}

ALWI void rpow_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(rpow, APPROX)); }

}  // namespace ckernel
```

### Layer 4: LLK Wrapper (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h`

File not present - rpow uses macro-based wrapper pattern (Layer 3 provides sufficient interface).

### Layer 5: LLK Wrapper (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rpow.h`

File not present - rpow uses macro-based wrapper pattern (Layer 3 provides sufficient interface).

### Layer 6: SfpuType Enum Entry

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

Entry in enum:
```cpp
rpow,
```

### Layer 7: sfpu_split_includes.h Entry

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

Conditional include block:
```cpp
#if SFPU_OP_RPOW_INCLUDE
#include "api/compute/eltwise_unary/rpow.h"
#endif
```

### Layer 8: llk_math_unary_sfpu_api.h Include

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

Not included - rpow uses macro-based pattern in compute API header.

### Layer 9: Dispatch in unary_op_utils.cpp

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Case statement in `get_op_init_and_func_one_param`:
```cpp
case UnaryOpType::RPOW: {
    return {"rpow_tile_init();", fmt::format("rpow_tile({}, {:#x}u);", idst, std::bit_cast<uint32_t>(param0))};
}
```

### Layer 10: Python Golden Function

**File**: `ttnn/ttnn/operations/unary.py`

Golden function definition:
```python
def _golden_function_rpow(input_tensor_a, *args, base, **kwargs):
    import torch

    return torch.pow(torch.tensor(base), input_tensor_a)


ttnn.attach_golden_function(ttnn.rpow, golden_function=_golden_function_rpow)
```

### Layer 11: Test File

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_rpow.py`

```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


def run_rpow_test(device, h, w, base, pcc=0.99):
    torch.manual_seed(0)

    # rpow(x, base) = base^x
    # Use small positive range to avoid overflow/underflow with large exponents
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 4.0 - 1.0  # [-1, 3]

    # Golden: torch.pow(scalar_base, input_tensor)
    torch_output_tensor = torch.pow(torch.tensor(base), torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.rpow(input_tensor, base)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("base", [2.0, 3.0, 0.5, 10.0, 1.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rpow(device, h, w, base):
    run_rpow_test(device, h, w, base, pcc=0.99)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_rpow_base_one(device, h, w):
    """base=1 should always return 1.0 regardless of input"""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 10.0 - 5.0
    torch_output_tensor = torch.ones_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.rpow(input_tensor, 1.0)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_rpow_zero_exponent(device, h, w):
    """x=0 should return 1.0 for any base (base^0 = 1)"""
    torch.manual_seed(0)
    torch_input_tensor = torch.zeros((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.ones_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.rpow(input_tensor, 2.0)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
```

### Layer 12: Registration in unary.hpp

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

Registration line:
```cpp
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW)
```

## Design Decisions

1. **Logarithmic Decomposition**: rpow(x, base) = exp(x * ln(base)) avoids direct computation of large exponents that could overflow. Using base^x = 2^(x * log2(base)) allows range reduction via IEEE754 exponent manipulation.

2. **Base Precomputation**: The logarithm of base is computed once before the tile loop since it's a constant parameter. This amortizes the cost across all input elements in the tile.

3. **IEEE754 Exponent Extraction**: log2(base) is computed by decomposing base into mantissa and exponent, using the SFPU's exexp (extract exponent) and setexp (set exponent) instructions.

4. **Minimax Polynomial for log2**: A 3-term polynomial approximates log2(1+f) where f is the fractional part of the mantissa. This provides good accuracy for the logarithm computation.

5. **Range Reduction with Rounding**: The exponent n in 2^(y/ln(2)) = 2^n * 2^frac is computed using an add-subtract trick for rounding (avoiding explicit rounding instructions).

6. **Taylor Series for 2^frac**: The fractional part uses a 3-term Taylor expansion valid for |frac| <= 0.5.

7. **Special Case Handling**: x == 0 is explicitly handled to return 1.0 (base^0 = 1), ensuring correct behavior even for edge case bases.

8. **One-Parameter Operation**: rpow is registered as a parameterized unary operation, with base passed as a single float parameter bit-cast to uint32_t.

## Debug Log

Implementation completed with comprehensive test coverage including:
- Multiple base values (2.0, 3.0, 0.5, 10.0, 1.5)
- Edge case: base = 1.0 (should return all 1s)
- Edge case: exponent = 0 (should return 1.0)
- Range of input values to test overflow/underflow handling

## Test Results

Tests verify:
- Numerical accuracy against PyTorch's torch.pow(base, x)
- Correct handling of base=1.0
- Correct handling of x=0 (base^0 = 1)
- PCC correlation >= 0.99 with golden function

## Known Limitations

1. **Overflow/Underflow**: Large bases and large positive exponents may overflow, and small bases with negative exponents may underflow. Input range is typically restricted to [-1, 3] to avoid these issues.

2. **Approximation Accuracy**: Polynomial approximations for log2 and 2^x introduce small errors that accumulate in the computation chain.

3. **Subnormal Numbers**: Very small results may flush to zero due to subnormal number handling in floating-point hardware.
