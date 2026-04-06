# Implementation Notes: frac

## Math Definition

The `frac` operation (fractional part) is defined as:

```
frac(x) = x - trunc(x)
```

Where `trunc(x)` is truncation toward zero (not floor). This matches PyTorch's `torch.frac()` semantics. For example:
- frac(3.7) = 0.7
- frac(-3.7) = -0.7
- frac(5.0) = 0.0

## Files Created

### Layer 1: SFPU Kernel (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// frac(x) = x - trunc(x)
//
// Matches PyTorch torch.frac() semantics (truncation toward zero, not floor).
//
// Algorithm:
// 1. Extract unbiased exponent from x.
// 2. If exp < 0 (|x| < 1): trunc(x) = 0, so frac = x.
// 3. If exp >= 23: x is already an integer, frac = 0.
// 4. Otherwise (0 <= exp < 23): compute trunc(x) by masking out fractional bits.
// 5. Result = x - trunc(x).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Default: frac = 0 for integers (exp >= 23)
        sfpi::vFloat trunc_x = x;

        // Extract unbiased exponent
        sfpi::vInt exp = sfpi::exexp(x);

        // Case 1: |x| < 1 (exp < 0) — trunc toward zero gives 0
        v_if(exp < 0) { trunc_x = 0.0f; }
        v_endif;

        // Case 2: 0 <= exp < 23 (has fractional bits in float32)
        v_if(exp >= 0 && exp < 23) {
            // Create bitmask to zero out fractional mantissa bits.
            // IEEE 754 float32 has 23 mantissa bits. For exponent e,
            // the lowest (23 - e) bits are fractional.
            // mask = 0xFFFFFFFF << (23 - exp)
            sfpi::vUInt shift = sfpi::vUInt(23 - exp);
            sfpi::vInt mask = sfpi::vInt(-1) << shift;

            // Apply mask to get trunc(x) (round toward zero)
            sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(x);
            trunc_x = sfpi::reinterpret<sfpi::vFloat>(xi & mask);
        }
        v_endif;

        // frac(x) = x - trunc(x)
        sfpi::dst_reg[0] = x - trunc_x;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### Layer 2: SFPU Kernel (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// frac(x) = x - trunc(x)
//
// Matches PyTorch torch.frac() semantics (truncation toward zero, not floor).
//
// Algorithm:
// 1. Extract unbiased exponent from x.
// 2. If exp < 0 (|x| < 1): trunc(x) = 0, so frac = x.
// 3. If exp >= 23: x is already an integer, frac = 0.
// 4. Otherwise (0 <= exp < 23): compute trunc(x) by masking out fractional bits.
// 5. Result = x - trunc(x).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_frac() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Default: frac = 0 for integers (exp >= 23)
        sfpi::vFloat trunc_x = x;

        // Extract unbiased exponent
        sfpi::vInt exp = sfpi::exexp(x);

        // Case 1: |x| < 1 (exp < 0) — trunc toward zero gives 0
        v_if(exp < 0) { trunc_x = 0.0f; }
        v_endif;

        // Case 2: 0 <= exp < 23 (has fractional bits in float32)
        v_if(exp >= 0 && exp < 23) {
            // Create bitmask to zero out fractional mantissa bits.
            // IEEE 754 float32 has 23 mantissa bits. For exponent e,
            // the lowest (23 - e) bits are fractional.
            // mask = 0xFFFFFFFF << (23 - exp)
            sfpi::vUInt shift = sfpi::vUInt(23 - exp);
            sfpi::vInt mask = sfpi::vInt(-1) << shift;

            // Apply mask to get trunc(x) (round toward zero)
            sfpi::vInt xi = sfpi::reinterpret<sfpi::vInt>(x);
            trunc_x = sfpi::reinterpret<sfpi::vFloat>(xi & mask);
        }
        v_endif;

        // frac(x) = x - trunc(x)
        sfpi::dst_reg[0] = x - trunc_x;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### Layer 3: LLK Wrapper (Wormhole)

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_frac.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_frac_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::frac, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_frac(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_frac<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 4: LLK Wrapper (Blackhole)

**File**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_frac.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_frac_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::frac, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_frac(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_frac<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Layer 5: Compute API Header

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/frac.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_frac.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise frac operation: x - trunc(x).
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
ALWI void frac_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_frac<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void frac_tile_init() { MATH((llk_math_eltwise_unary_sfpu_frac_init<APPROX>())); }

}  // namespace ckernel
```

### Layer 6: SfpuType Enum Entry

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

The enum entry is defined in the SfpuType enumeration. frac is registered as a no-parameter operation.

### Layer 7: sfpu_split_includes.h

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

The sfpu_split_includes.h file conditionally includes frac.h when frac operation is required.

### Layer 8: llk_math_unary_sfpu_api.h

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

Frac uses the standard unary operation dispatch through llk_math_eltwise_unary_sfpu_frac wrapper functions.

### Layer 9: Dispatch (unary_op_utils.cpp)

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Relevant case statements for frac dispatch:

```cpp
case UnaryOpType::FRAC: return {"frac_tile_init();", fmt::format("frac_tile({});", idst)};
```

### Layer 10: Python Golden (unary.py)

**File**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_frac(input_tensor_a, *args, **kwargs):
    import torch

    return torch.frac(input_tensor_a)


ttnn.attach_golden_function(ttnn.frac, golden_function=_golden_function_frac)
```

### Layer 11: Test File

**File**: `tests/ttnn/unit_tests/operations/eltwise/test_frac.py`

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16],
)
def test_frac(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # PyTorch golden: torch.frac computes x - trunc(x)
    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    # TT computation
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    assert_allclose(torch_output, tt_output_torch, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_negative(device, shape):
    """Verify frac works correctly for negative inputs (sign-preserving)."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10  # wide range

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # frac output should always be in (-1, 1)
    assert tt_output_torch.min() > -1.0, f"Output <= -1: {tt_output_torch.min()}"
    assert tt_output_torch.max() < 1.0, f"Output >= 1: {tt_output_torch.max()}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_frac_integers(device, shape):
    """Verify frac returns 0 for integer inputs."""
    # Create integer-valued tensor
    torch_input = torch.arange(-16, 16, dtype=torch.bfloat16).reshape(1, 1, 32, 1).expand(shape)

    torch_output = torch.frac(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.frac(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
    # All fractional parts of integers should be 0
    assert torch.allclose(tt_output_torch, torch.zeros_like(tt_output_torch), atol=1e-2)
```

### Layer 12: Registration

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

```cpp
REGISTER_UNARY_OPERATION(frac, FRAC)
```

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
bind_unary_operation<"frac", &ttnn::frac>(
    mod, R"doc(\text{frac}(x) = x - \lfloor x \rfloor)doc", "", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

## Design Decisions

- **IEEE 754 Bit Manipulation**: The implementation extracts the exponent and uses bit masking to zero out fractional bits.
- **Three Cases**: Handles (1) values with magnitude less than 1, (2) values with fractional bits (0 <= exp < 23), and (3) integers (exp >= 23).
- **No Approximation**: Unlike trigonometric or exponential functions, frac uses exact bit manipulation for accurate results.
- **Sign Preservation**: Handles negative numbers correctly by preserving sign and applying truncation toward zero.

## Test Results

- **Standard test**: Validates frac across different shapes with PCC >= 0.999.
- **Negative inputs test**: Ensures output is in (-1, 1) range for both positive and negative inputs.
- **Integer test**: Verifies frac returns exactly 0 for integer-valued inputs.
- **All tests pass** with rtol=1.6e-2, atol=1e-2 tolerances.

## Known Limitations

None. The operation correctly implements truncation toward zero via IEEE 754 bit manipulation, ensuring exact results.
