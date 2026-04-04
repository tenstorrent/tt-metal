# Implementation Notes: hardswish

## Math Definition

hardswish(x) = x * min(max(x + 3, 0), 6) / 6 = x * hardsigmoid(x)

Where hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)

## Implementation Strategy

Hardswish is mathematically `x * hardsigmoid(x)`, where `hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)`. The SFPU kernel computes the hardsigmoid intermediate, then multiplies by the original input `x`. This avoids needing any sub-function calls (exp, reciprocal, etc.) and keeps the kernel entirely self-contained with simple arithmetic and clamping.

## Reference Operations Used

- **hardsigmoid** (most useful): The kernel is a direct extension of hardsigmoid. The ckernel, LLK dispatch, and API header files all follow the exact same structure. The only difference is the final store: hardsigmoid stores `result` directly, while hardswish stores `x * result`.
- **hardtanh**: Provided a secondary pattern for clamping with `v_if`/`v_endif` blocks.
- **softsign** and **selu**: Referenced for understanding the full abstraction layer structure (API header macros, LLK dispatch patterns, params dispatch).

## Design Decisions

1. **Piecewise computation via clamping**: Rather than computing separate branches for x <= -3, -3 < x < 3, and x >= 3, we compute the continuous function and clamp the intermediate hardsigmoid to [0, 1]. This is simpler and avoids branching complexity.

2. **No special init function**: Unlike some operations, hardswish doesn't need any programmable LUT initialization or constant loading. All constants (1/6 and 0.5) are compile-time constants in the ckernel.

3. **Template parameter APPROXIMATION_MODE**: The kernel accepts this parameter for consistency with the LLK interface, but it's not used since the computation is already simple arithmetic.

4. **Full unrolling with #pragma GCC unroll 8**: Matches the pattern used by hardsigmoid and other similar ops, processing 8 iterations per loop.

## Deviations from Standard Patterns

None. The implementation follows the exact same pattern as hardsigmoid across all layers:
- No init function needed (no programmable constants)
- No parameters (non-parameterized op)
- Standard `VectorMode::RC` dispatch
- `ADDR_MOD_7` with all-zero increments (default)
- `APPROXIMATION_MODE` template parameter accepted but unused

## Files Created

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`

SFPU kernel implementing the hardswish computation. Computes hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1) and multiplies by x.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// hardswish(x) = x * min(max(x + 3, 0), 6) / 6
//              = x * hardsigmoid(x)
//              = x * clamp(x/6 + 0.5, 0, 1)
// Piecewise:
//   x <= -3  =>  0
//   x >= 3   =>  x
//   else     =>  x * (x/6 + 0.5)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardswish() {
    constexpr float one_sixth = 1.0f / 6.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat hsigmoid = x * one_sixth + 0.5f;

        // Clamp hardsigmoid to [0, 1]
        v_if(hsigmoid < 0.0f) { hsigmoid = 0.0f; }
        v_endif;
        v_if(hsigmoid > sfpi::vConst1) { hsigmoid = sfpi::vConst1; }
        v_endif;

        sfpi::dst_reg[0] = x * hsigmoid;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardswish.h`.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// hardswish(x) = x * min(max(x + 3, 0), 6) / 6
//              = x * hardsigmoid(x)
//              = x * clamp(x/6 + 0.5, 0, 1)
// Piecewise:
//   x <= -3  =>  0
//   x >= 3   =>  x
//   else     =>  x * (x/6 + 0.5)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardswish() {
    constexpr float one_sixth = 1.0f / 6.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat hsigmoid = x * one_sixth + 0.5f;

        // Clamp hardsigmoid to [0, 1]
        v_if(hsigmoid < 0.0f) { hsigmoid = 0.0f; }
        v_endif;
        v_if(hsigmoid > sfpi::vConst1) { hsigmoid = sfpi::vConst1; }
        v_endif;

        sfpi::dst_reg[0] = x * hsigmoid;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`

LLK dispatch header that wraps the ckernel calculation and initializes the math engine.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_hardswish.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardswish_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardswish(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardswish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`

Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardswish.h`.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_hardswish.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_hardswish_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::hardswish, APPROXIMATE>();
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_hardswish(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardswish<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/hardswish.h`

High-level compute API header exposing hardswish_tile() and hardswish_tile_init() functions to compute kernels.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_hardswish.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise hardswish operation: x * min(max(x + 3, 0), 6) / 6.
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
ALWI void hardswish_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_hardswish<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void hardswish_tile_init() { MATH((llk_math_eltwise_unary_sfpu_hardswish_init<APPROX>())); }

}  // namespace ckernel
```

### `tests/ttnn/unit_tests/operations/eltwise/test_hardswish.py`

Unit tests for the hardswish operation covering basic functionality and piecewise behavior verification.

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


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
def test_hardswish(device, shape, dtype):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    # PyTorch golden
    torch_output = torch.nn.functional.hardswish(torch_input.float()).to(torch.bfloat16)

    # TT computation
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardswish(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_hardswish_piecewise(device, shape):
    """Verify hardswish piecewise behavior: 0 for x<=-3, x for x>=3, and smooth in between."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 10  # wide range

    torch_output = torch.nn.functional.hardswish(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.hardswish(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)

    # For x <= -3, hardswish(x) == 0
    mask_neg = torch_input <= -3.0
    if mask_neg.any():
        assert (tt_output_torch[mask_neg] == 0.0).all(), "hardswish should be 0 for x <= -3"

    # For x >= 3, hardswish(x) == x
    mask_pos = torch_input >= 3.0
    if mask_pos.any():
        assert_with_pcc(torch_input[mask_pos], tt_output_torch[mask_pos], pcc=0.999)
```

## Files Modified

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

Added hardswish to the SfpuType enum.

```diff
@@ -15,4 +15,5 @@ enum class SfpuType {
     lgamma,
     rpow,
     silu,
+    hardswish,
 };
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`

Added hardswish to the SfpuType enum (identical change to wormhole_b0 version).

```diff
@@ -15,4 +15,5 @@ enum class SfpuType {
     lgamma,
     rpow,
     silu,
+    hardswish,
 };
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

Added conditional include guard for the hardswish compute API header.

```diff
@@ -39,3 +39,7 @@
 #if SFPU_OP_RPOW_INCLUDE
 #include "api/compute/eltwise_unary/rpow.h"
 #endif
+
+#if SFPU_OP_HARDSWISH_INCLUDE
+#include "api/compute/eltwise_unary/hardswish.h"
+#endif
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

Added hardswish to the macro definition mapping and operation init/function pair mapping.

```diff
@@ -25,6 +25,7 @@ std::string get_macro_definition(UnaryOpType op_type) {
         case UnaryOpType::SOFTSIGN: return "SFPU_OP_SOFTSIGN_INCLUDE";
         case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
         case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
+        case UnaryOpType::HARDSWISH: return "SFPU_OP_HARDSWISH_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     };
 }
@@ -71,6 +72,7 @@ std::pair<std::string, std::string> get_op_init_and_func_default(
         case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
         case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
         case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
+        case UnaryOpType::HARDSWISH: return {"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)};
         case UnaryOpType::SELU: return {"selu_tile_init();", fmt::format("selu_tile({});", idst)};
         case UnaryOpType::ATANH: return {"atanh_tile_init();", fmt::format("atanh_tile({});", idst)};
         case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
```

### `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`

Added hardswish to the macro definition mapping and updated the operation init/function pair mapping (changed from empty return to actual implementation).

```diff
@@ -22,6 +22,7 @@ std::string get_macro_definition(UnaryOpType op_type) {
         case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";
         case UnaryOpType::ATANH: return "SFPU_OP_ATANH_INCLUDE";
         case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
+        case UnaryOpType::HARDSWISH: return "SFPU_OP_HARDSWISH_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     }
 }
@@ -89,7 +90,7 @@ std::pair<std::string, std::string> get_op_init_and_func(
         case UnaryOpType::TRUNC: return {"rounding_op_tile_init();", fmt::format("trunc_tile({});", idst)};
         case UnaryOpType::FRAC: return {"rounding_op_tile_init();", fmt::format("frac_tile({});", idst)};
         case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
-        case UnaryOpType::HARDSWISH: return {};
+        case UnaryOpType::HARDSWISH: return {"hardswish_tile_init();", fmt::format("hardswish_tile({});", idst)};
         case UnaryOpType::LGAMMA: return {"lgamma_tile_init();", fmt::format("lgamma_tile({});", idst)};
         case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
         case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
```

### `ttnn/ttnn/operations/unary.py`

Added golden function for hardswish using PyTorch's implementation.

```diff
@@ -108,6 +108,15 @@ def _golden_function_rpow(input_tensor_a, base, *args, **kwargs):
 ttnn.attach_golden_function(ttnn.rpow, golden_function=_golden_function_rpow)


+def _golden_function_hardswish(input_tensor_a, *args, **kwargs):
+    import torch
+
+    return torch.nn.functional.hardswish(input_tensor_a)
+
+
+ttnn.attach_golden_function(ttnn.hardswish, golden_function=_golden_function_hardswish)
+
+
 try:
     SigmoidMode = ttnn._ttnn.operations.unary.SigmoidMode
 except AttributeError:
```

## Test Results

All tests passed successfully:
- `test_hardswish[dtype=DataType.BFLOAT16-shape=[1, 1, 32, 32]]` -- PASS
- `test_hardswish[dtype=DataType.BFLOAT16-shape=[1, 1, 320, 384]]` -- PASS
- `test_hardswish[dtype=DataType.BFLOAT16-shape=[1, 3, 320, 384]]` -- PASS
- `test_hardswish_piecewise[shape=[1, 1, 32, 32]]` -- PASS

## Known Limitations

- The operation inherits bfloat16 precision limitations from the hardsigmoid computation (the `x/6 + 0.5` intermediate).
- For very large `|x|`, the clamping ensures correct piecewise behavior (0 for x <= -3, x for x >= 3), but precision may be affected by bfloat16 representation.
- Only tested with bfloat16 dtype in the unit tests; float32 compatibility is assumed but not explicitly tested in this test suite.
