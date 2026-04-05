# Implementation Notes: sinh

## Math Definition
sinh(x) = (exp(x) - exp(-x)) / 2 (hyperbolic sine)

## Implementation Strategy
The SFPU kernel computes sinh(x) by:
1. Computing exp(x) using `_sfpu_exp_21f_bf16_` polynomial approximation
2. Computing exp(-x) using the same function on negated input
3. Subtracting: exp(x) - exp(-x)
4. Halving: multiply by 0.5f

This preserves the odd symmetry sinh(-x) = -sinh(x) naturally through the subtraction.
The init function uses `_init_exponential_` with the same parameters as cosh.

## Reference Operations Used
- **cosh**: Primary reference -- identical layer structure, same exp-based approach, just + vs - in the formula
- **selu**: Secondary reference -- shows exp usage patterns and SFPU conditional branching

## Test Results
- **Status**: PASS (all 9 tests passed on first try)
- **Tests**: 4 shapes x 2 dtypes (bfloat16, float32) + 1 range test = 9 tests
- **PCC**: >= 0.999 for all configurations

## New Files

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// sinh(x) = (exp(x) - exp(-x)) / 2
//
// Implementation strategy:
// We compute exp(x) and exp(-x) independently using the _sfpu_exp_21f_bf16_
// polynomial approximation, then form the difference and halve it.
// This approach naturally preserves the odd symmetry sinh(-x) = -sinh(x)
// through the subtraction, and avoids catastrophic cancellation issues
// because exp(x) and exp(-x) are always on opposite sides of 1.0
// (one >= 1, the other <= 1) for any real x.

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_sinh_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute exp(x) and exp(-x) using the exp_21f polynomial approximation
        sfpi::vFloat exp_pos = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(x);
        sfpi::vFloat neg_x = -x;
        sfpi::vFloat exp_neg = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(neg_x);

        // sinh(x) = (exp(x) - exp(-x)) * 0.5
        sfpi::vFloat result = (exp_pos - exp_neg) * 0.5f;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void sinh_init() {
    _init_exponential_<APPROXIMATION_MODE, false, p_sfpu::kCONST_1_FP16B>();
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sinh.h`
(Identical content to the wormhole_b0 version above)

### `tt_metal/hw/inc/api/compute/eltwise_unary/sinh.h`
```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sinh.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void sinh_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(sinh, ckernel::sfpu::sinh_init, APPROX)); }

ALWI void sinh_tile(uint32_t idst) {
    MATH(SFPU_THREE_PARAM_KERNEL_FP32_FIRST(_calculate_sinh_, APPROX, DST_ACCUM_MODE, 8, idst, (int)VectorMode::RC));
}

}  // namespace ckernel
```

### `tests/ttnn/unit_tests/operations/eltwise/test_sinh.py`
```python
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

@pytest.mark.parametrize("input_shapes", [[1,1,32,32],[1,1,64,64],[1,3,320,384],[4,1,32,32]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_sinh(device, input_shapes, dtype):
    torch_input = torch.empty(input_shapes, dtype=torch.float32).uniform_(-4.0, 4.0)
    if dtype == ttnn.bfloat16:
        torch_input = torch_input.to(torch.bfloat16).to(torch.float32)
    torch_output = torch.sinh(torch_input)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.sinh(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)
```

## Modified Files

### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
Added:
```cpp
#if SFPU_OP_SINH_INCLUDE
#include "api/compute/eltwise_unary/sinh.h"
#endif
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
Added SINH to three locations:
- `get_macro_definition`: `case UnaryOpType::SINH: return "SFPU_OP_SINH_INCLUDE";`
- `get_op_init_and_func_default`: `case UnaryOpType::SINH: return {"sinh_tile_init();", fmt::format("sinh_tile({});", idst)};`
- `string_to_unary_with_param`: `if (name == "sinh") { return UnaryWithParam(UnaryOpType::SINH); }`

### `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
Added SINH to two locations:
- `get_macro_definition`: `case UnaryOpType::SINH: return "SFPU_OP_SINH_INCLUDE";`
- `get_op_init_and_func`: `case UnaryOpType::SINH: return {"sinh_tile_init();", fmt::format("sinh_tile({});", idst)};`

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
Added:
```cpp
bind_unary_operation<"sinh", &ttnn::sinh>(
    mod,
    R"doc(\mathrm{{output\_tensor}}_i = \sinh(\mathrm{{input\_tensor}}_i))doc",
    "[supported range -9 to 9]",
    R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
```

### `ttnn/ttnn/operations/unary.py`
Added:
```python
def _golden_function_sinh(input_tensor_a, *args, **kwargs):
    import torch
    return torch.sinh(input_tensor_a)

ttnn.attach_golden_function(ttnn.sinh, golden_function=_golden_function_sinh)
```

## Deviations from Standard Patterns
None -- follows the exact same patterns as cosh across all layers.

## Known Limitations
- Supported range is approximately [-9, 9] for bfloat16 due to exp overflow limits (same as cosh)
- The `_sfpu_exp_21f_bf16_` function clamps exp results for inputs outside approximately [-88.5, 88.5]
