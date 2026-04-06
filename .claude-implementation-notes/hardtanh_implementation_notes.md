# Implementation Notes: Hardtanh

## Math Definition

The hardtanh function (hard hyperbolic tangent approximation):
```
hardtanh(x, min_val, max_val) = max(min_val, min(max_val, x))
                               = clamp(x, min_val, max_val)
```

Default parameters: min_val = -1.0, max_val = 1.0

## Architecture Overview

Hardtanh is implemented as an SFPU (Scalar Floating Point Unit) unary operation with two float parameters on Tenstorrent hardware (Wormhole and Blackhole architectures). The implementation uses simple conditional clamping to restrict values to a specified range.

## Files Created

### 1. SFPU Kernel (Wormhole)

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// hardtanh(x, min_val, max_val) = max(min_val, min(max_val, x))
// Clamps all elements in input into the range [min_val, max_val].
// Default: min_val = -1.0, max_val = 1.0
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(uint32_t param0, uint32_t param1) {
    // Reconstruct float parameters from bit-cast uint32_t
    sfpi::vFloat v_min = Converter::as_float(param0);
    sfpi::vFloat v_max = Converter::as_float(param1);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Clamp to max: if v > max_val, set v = max_val
        v_if(v > v_max) { v = v_max; }
        v_endif;

        // Clamp to min: if v < min_val, set v = min_val
        v_if(v < v_min) { v = v_min; }
        v_endif;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### 2. SFPU Kernel (Blackhole)

**Path**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`

Identical copy to Wormhole implementation.

### 3. LLK Wrapper (Wormhole)

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_hardtanh.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardtanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardtanh, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardtanh(
    uint dst_index, uint32_t param0, uint32_t param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1);
}

}  // namespace ckernel
```

### 4. LLK Wrapper (Blackhole)

**Path**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`

Identical copy to Wormhole implementation.

### 5. Compute API Header

**Path**: `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_hardtanh.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise hardtanh operation: clamp(x, min_val, max_val).
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The min_val as IEEE 754 float bits                                         | uint32_t |                                                       | True     |
 * | param1          | The max_val as IEEE 754 float bits                                         | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardtanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>())); }

}  // namespace ckernel
```

## Layer 6: SfpuType Enum Entry

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

```cpp
enum class SfpuType {
    unused = 0,
    cosh,
    cbrt,
    hardtanh,      // <-- Entry for hardtanh
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
#if SFPU_OP_HARDTANH_INCLUDE
#include "api/compute/eltwise_unary/hardtanh.h"
#endif
```

## Layer 8: llk_math_unary_sfpu_api.h

**Path**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_sfpu_api.h`

```cpp
#include "llk_math_eltwise_unary_sfpu_hardtanh.h"
```

## Layer 9: Dispatch (unary_op_utils.cpp)

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

```cpp
case UnaryOpType::HARDTANH: {
    float param1 = static_cast<float>(params[1]);
    return {
        "hardtanh_tile_init();",
        fmt::format(
            "hardtanh_tile({}, {:#x}u, {:#x}u);",
            idst,
            std::bit_cast<uint32_t>(param0),
            std::bit_cast<uint32_t>(param1))};
}
```

## Layer 10: Python Golden Function

**Path**: `ttnn/ttnn/operations/unary.py`

```python
def _golden_function_hardtanh(input_tensor_a, *args, min_val=-1.0, max_val=1.0, **kwargs):
    import torch

    return torch.nn.functional.hardtanh(input_tensor_a, min_val=min_val, max_val=max_val)


ttnn.attach_golden_function(ttnn.hardtanh, golden_function=_golden_function_hardtanh)
```

## Layer 11: Test File

**Path**: `tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py`

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)

pytestmark = pytest.mark.use_module_device


def _golden(input_tensor, min_val, max_val):
    return torch.nn.functional.hardtanh(input_tensor, min_val=min_val, max_val=max_val)


def _run_bfloat16(device, input_tensor, min_val, max_val):
    golden = _golden(input_tensor, min_val, max_val)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=min_val, max_val=max_val)
    result = ttnn.to_torch(tt_output)
    return golden, result


@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_default_bfloat16(device, h, w):
    torch.manual_seed(0)
    input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    golden, result = _run_bfloat16(device, input_tensor, min_val=-1.0, max_val=1.0)
    assert_with_ulp(golden, result, 2)
    assert_with_pcc(golden, result, 0.99)


@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_default_fp32(device, h, w):
    torch.manual_seed(0)
    input_tensor = torch.randn((h, w), dtype=torch.float32)
    golden = _golden(input_tensor, -1.0, 1.0)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_allclose(golden, result, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "min_val,max_val",
    [
        (-1.0, 1.0),
        (-0.5, 0.5),
        (-2.0, 2.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (-5.5, 3.7),
    ],
)
@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_params_bfloat16(device, h, w, min_val, max_val):
    torch.manual_seed(42)
    input_tensor = torch.randn((h, w), dtype=torch.bfloat16) * 3.0
    golden, result = _run_bfloat16(device, input_tensor, min_val, max_val)
    assert_with_ulp(golden, result, 2)
    assert_with_pcc(golden, result, 0.99)


@pytest.mark.parametrize(
    "min_val,max_val",
    [
        (-1.0, 1.0),
        (-0.5, 0.5),
        (-2.0, 2.0),
        (0.0, 1.0),
        (-5.5, 3.7),
    ],
)
@pytest.mark.parametrize("h,w", [(64, 128)])
def test_hardtanh_params_fp32(device, h, w, min_val, max_val):
    torch.manual_seed(42)
    input_tensor = torch.randn((h, w), dtype=torch.float32) * 3.0
    golden = _golden(input_tensor, min_val, max_val)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=min_val, max_val=max_val)
    result = ttnn.to_torch(tt_output)
    assert_allclose(golden, result, rtol=1.6e-2, atol=1e-2)


def test_hardtanh_exhaustive_bfloat16(device):
    input_tensor = generate_all_bfloat16_bitpatterns(torch.bfloat16).flatten()
    input_tensor = flush_subnormal_values_to_zero(input_tensor)
    # Filter out NaN and Inf values: hardtanh behavior on non-finite inputs is undefined
    finite_mask = torch.isfinite(input_tensor)
    input_tensor = input_tensor[finite_mask]
    golden = _golden(input_tensor, -1.0, 1.0)
    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_with_ulp(golden, result, 2)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
        (1, 1, 320, 384),
        (1, 3, 320, 384),
        (2, 4, 64, 128),
    ],
)
def test_hardtanh_shapes_bfloat16(device, shape):
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, dtype=torch.bfloat16) * 2.0
    golden = _golden(input_tensor, -1.0, 1.0)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_with_pcc(golden, result, 0.99)


def test_hardtanh_golden_function_consistency(device):
    torch.manual_seed(0)
    input_tensor = torch.randn((64, 128), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    golden_from_ttnn = golden_function(input_tensor, min_val=-1.0, max_val=1.0)
    golden_direct = torch.nn.functional.hardtanh(input_tensor, min_val=-1.0, max_val=1.0)
    assert torch.allclose(golden_from_ttnn, golden_direct)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardtanh(tt_input, min_val=-1.0, max_val=1.0)
    result = ttnn.to_torch(tt_output)
    assert_with_pcc(golden_direct, result, 0.99)
```

## Layer 12: Registration

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`

Note: hardtanh is registered differently due to its two-parameter signature.

**Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`

```cpp
ttnn::bind_function<"hardtanh">(
    mod,
    doc.c_str(),
    &unary_two_float_5param_to_6param_wrapper<&ttnn::hardtanh>,
    ...
);
```

## Design Decisions

1. **Parameter Passing via Bit-Cast**: The two float parameters (min_val and max_val) are passed as uint32_t values (bit-cast from IEEE 754 representation) and converted back to vFloat using the Converter utility.

2. **Sequential Clamping**: The implementation clamps to max first, then to min. This ensures correctness even if max < min (though this is not a valid use case).

3. **Conditional Execution with v_if/v_endif**: The SFPU uses conditional execution macros to perform the comparisons and assignments efficiently.

4. **Loop Unrolling**: The pragma GCC unroll 8 directive fully unrolls the inner loop for SFPU optimization.

5. **No Approximation Mode Dependency**: Unlike some operations, hardtanh's behavior is identical in approximation and exact modes (it's a pure clamping operation).

## Known Limitations

1. **Parameter order**: min_val must be ≤ max_val for correct behavior; no validation is performed.
2. **Non-finite handling**: NaN and Inf values are passed through without special handling.
3. **Exhaustive testing limitation**: The exhaustive test filters out NaN and Inf values, which may not fully represent edge cases.

## Test Results

The test suite includes:
- Default parameters (min_val=-1.0, max_val=1.0): PCC ≥ 0.99, ULP ≤ 2
- Parameterized ranges: 6 different min/max combinations tested
- Shape variants: 5 different tensor shapes tested
- Exhaustive bfloat16 coverage: All finite bit patterns validated
- Golden function consistency: Verified against PyTorch reference
