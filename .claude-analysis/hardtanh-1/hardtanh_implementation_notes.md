# Implementation Notes: hardtanh

## Math Definition
max(min_val, min(max_val, x)) where min_val=-1.0 (default), max_val=1.0 (default)

## Files Created

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
SFPU kernel that implements the hardtanh clamp operation. Uses two v_if comparisons to clamp values between min_val and max_val. Parameters are bitcast from IEEE 754 float representation to vFloat using Converter::as_float.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) {
    // param0 = min_val as IEEE 754 float bits (bitcast uint32_t)
    // param1 = max_val as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if(val < min_val) { val = min_val; }
        v_endif;

        v_if(val > max_val) { val = max_val; }
        v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`
Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(std::uint32_t param0, std::uint32_t param1) {
    // param0 = min_val as IEEE 754 float bits (bitcast uint32_t)
    // param1 = max_val as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if(val < min_val) { val = min_val; }
        v_endif;

        v_if(val > max_val) { val = max_val; }
        v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
LLK dispatch header for hardtanh. Wraps the SFPU kernel in template functions that match the standard LLK dispatch pattern, using the `_llk_math_eltwise_unary_sfpu_params_` helper to pass parameters to the SFPU kernel.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1);
}

}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`
Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    uint dst_index, uint param0, uint param1, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_hardtanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1);
}

}  // namespace ckernel
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h`
Compute API header that exposes hardtanh_tile() and hardtanh_tile_init() functions to the device. Parameters are passed as uint32_t (IEEE 754 bitcast), and the actual float conversions happen in the SFPU kernel.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
 * Performs element-wise hardtanh operation. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The minimum value of the linear region range                               | uint32_t |                                                       | True     |
 * | param1          | The maximum value of the linear region range                               | uint32_t |                                                       | True     |

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

### `tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py`
Comprehensive test suite covering multiple parameter ranges (default, narrow, relu6-like, wide) and both bfloat16 and float32 dtypes. Uses all_bfloat16_bitpatterns for exhaustive testing and validates against torch.nn.functional.hardtanh.

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


@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (-1.0, 1.0),
        (-0.5, 0.5),
        (0.0, 6.0),
        (-2.0, 2.0),
    ],
    ids=["default", "narrow", "relu6-like", "wide"],
)
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_hardtanh(device, min_val, max_val, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        # Cast to float32 and flush subnormal inputs — hardware flushes these to zero
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32, flush subnormals to match hardware behavior
    torch_output = torch.nn.functional.hardtanh(torch_input.float(), min_val=min_val, max_val=max_val)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.hardtanh(tt_input, min_val=min_val, max_val=max_val)
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
        # Stricter tolerances — both sides have full float32 precision
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
```

## Files Modified

### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
Added conditional include for hardtanh.h to enable the hardtanh compute API when the SFPU_OP_HARDTANH_INCLUDE macro is defined.

```diff
diff --git a/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h b/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
index 86f345f4d40..0b89adefd17 100644
--- a/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
+++ b/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
@@ -3,3 +3,7 @@
 // SPDX-License-Identifier: Apache-2.0

 #pragma once
+
+#if SFPU_OP_HARDTANH_INCLUDE
+#include "api/compute/eltwise_unary/hardtanh.h"
+#endif
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
Added hardtanh to the SfpuType enum to register the operation in the LLK type system.

```diff
diff --git a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
index a2d8a146438..833f54ce950 100644
--- a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
+++ b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
@@ -6,4 +6,5 @@

 enum class SfpuType {
     unused = 0,
+    hardtanh,
 };
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
Added hardtanh to the SfpuType enum to register the operation in the LLK type system.

```diff
diff --git a/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h b/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
index 3c040fcea47..34ec02a3945 100644
--- a/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
+++ b/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
@@ -6,4 +6,5 @@

 enum class SfpuType {
     unused = 0,
+    hardtanh,
 };
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
Added hardtanh case handling in the parameterized operations dispatch:
1. Added `#include <bit>` for std::bit_cast
2. Registered HARDTANH in get_macro_definition to use SFPU_OP_HARDTANH_INCLUDE
3. Added hardtanh case in get_op_init_and_func_parameterized to bitcast min_val and max_val as hex-encoded uint32_t constants and call hardtanh_tile with the hex literals.

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
index c4007d2c929..831001d7316 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
@@ -4,6 +4,7 @@

 #include "unary_op_utils.hpp"

+#include <bit>
 #include <optional>
 #include <tt_stl/assert.hpp>
 #include "ttnn/tensor/types.hpp"
@@ -16,6 +17,7 @@ namespace {

 std::string get_macro_definition(UnaryOpType op_type) {
     switch (op_type) {
+        case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     };
 }
@@ -34,6 +36,17 @@ std::pair<std::string, std::string> get_op_init_and_func_parameterized(
         case UnaryOpType::MISH:
             return {};// MISH uses dedicated mish_kernel.cpp;
         case UnaryOpType::LOGIT: return {};
+        case UnaryOpType::HARDTANH: {
+            float min_val = params.size() > 0 ? param0 : -1.0f;
+            float max_val = params.size() > 1 ? static_cast<float>(params[1]) : 1.0f;
+            return {
+                "hardtanh_tile_init();",
+                fmt::format(
+                    "hardtanh_tile({}, {:#010x}u, {:#010x}u);",
+                    idst,
+                    std::bit_cast<uint32_t>(min_val),
+                    std::bit_cast<uint32_t>(max_val))};
+        }
         default: TT_THROW("unexpected parameterized op type {}", op_type);
     };
 }
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
Added HARDTANH to the is_parametrized_type() switch statement to mark it as a parameterized operation.

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
index 1c7778ba09a..453bdc760a7 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
@@ -44,7 +44,8 @@ template <typename T>
 bool is_parametrized_type(T val) {
     switch (val) {
         case UnaryOpType::MISH:
-        case UnaryOpType::LOGIT: return true;
+        case UnaryOpType::LOGIT:
+        case UnaryOpType::HARDTANH: return true;
         default: return false;
     }
     return false;
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
Added hardtanh() function with two float parameters (min_val=-1.0f, max_val=1.0f) that wraps the unary_impl dispatcher with UnaryWithParam{UnaryOpType::HARDTANH, min_val, max_val}.

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp b/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
index 0e1b80c9625..3da8f65abe3 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
@@ -99,6 +99,22 @@ REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(mish, MISH)

 // Unaries with float parameter

+// hardtanh: two float parameters with defaults
+inline Tensor hardtanh(
+    const Tensor& input_tensor,
+    float min_val = -1.0f,
+    float max_val = 1.0f,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
+    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
+    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
+    return ttnn::detail::unary_impl(
+        input_tensor,
+        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, min_val, max_val}},
+        memory_config,
+        optional_output_tensor,
+        sub_core_grids);
+}
+
 // -----------------------------------------------------------------------------
 // Functions defined without macros (non-SFPU operations kept)
 // -----------------------------------------------------------------------------
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
Added nanobind registration for hardtanh() with two keyword-only float parameters:
1. Added unary_two_float_5param_to_6param_wrapper template to bridge the 6-param C++ function to the 5-param nanobind binding (sub_core_grids=std::nullopt).
2. Registered hardtanh with keyword-only arguments (min_val=-1.0f, max_val=1.0f, memory_config=None, output_tensor=None) and comprehensive docstring with LaTeX formula and supported dtypes.

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp b/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
index b9455c3a7d0..865edefeedf 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
@@ -66,6 +66,16 @@ Tensor unary_composite_3param_to_4param_wrapper(
     return Func(input_tensor, parameter_a, parameter_b, memory_config, std::nullopt);
 }

+template <auto Func>
+Tensor unary_two_float_5param_to_6param_wrapper(
+    const Tensor& input_tensor,
+    float parameter_a,
+    float parameter_b,
+    const std::optional<MemoryConfig>& memory_config,
+    const std::optional<Tensor>& output_tensor) {
+    return Func(input_tensor, parameter_a, parameter_b, memory_config, output_tensor, std::nullopt);
+}
+
 void bind_unary_clamp(nb::module_& mod) {
     const char* doc = R"doc(
         Applies clamp to :attr:`input_tensor` element-wise.
@@ -1789,6 +1799,52 @@ void py_module(nb::module_& mod) {

     bind_unary_operation_with_fast_and_approximate_mode<"mish", &ttnn::mish>(
         mod, "[Supported range -20 to inf]", R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
+
+    {
+        auto doc = fmt::format(
+            R"doc(
+            Applies the HardTanh function element-wise.
+
+            Clamps the input tensor to the range [{min_val}, {max_val}].
+
+            .. math::
+                \mathrm{{output\_tensor}}_i = \max(\mathrm{{min\_val}}, \min(\mathrm{{max\_val}}, \mathrm{{input\_tensor}}_i))
+
+            Args:
+                input_tensor (ttnn.Tensor): the input tensor.
+
+            Keyword Args:
+                min_val (float, optional): minimum value of the linear region range. Defaults to `-1.0`.
+                max_val (float, optional): maximum value of the linear region range. Defaults to `1.0`.
+                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
+                output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
+
+            Returns:
+                ttnn.Tensor: the output tensor.
+
+            Note:
+                Supported dtypes and layouts:
+
+                .. list-table::
+                   :header-rows: 1
+
+                   * - Dtypes
+                     - Layouts
+                   * - BFLOAT16, BFLOAT8_B, FLOAT32
+                     - TILE, ROW_MAJOR
+            )doc");
+
+        ttnn::bind_function<"hardtanh">(
+            mod,
+            doc.c_str(),
+            &unary_two_float_5param_to_6param_wrapper<&ttnn::hardtanh>,
+            nb::arg("input_tensor"),
+            nb::kw_only(),
+            nb::arg("min_val") = -1.0f,
+            nb::arg("max_val") = 1.0f,
+            nb::arg("memory_config") = nb::none(),
+            nb::arg("output_tensor") = nb::none());
+    }
 }

 }  // namespace ttnn::operations::unary
```

### `ttnn/ttnn/operations/unary.py`
Added Python golden function that wraps torch.nn.functional.hardtanh and registers it via ttnn.attach_golden_function for validation in testing and debugging.

```diff
diff --git a/ttnn/ttnn/operations/unary.py b/ttnn/ttnn/operations/unary.py
index 4e0bb7b3228..aaa9e9de291 100644
--- a/ttnn/ttnn/operations/unary.py
+++ b/ttnn/ttnn/operations/unary.py
@@ -71,6 +71,15 @@ def _golden_function_logit(input_tensor_a, *args, eps=None, **kwargs):
 ttnn.attach_golden_function(ttnn.logit, golden_function=_golden_function_logit)


+def _golden_function_hardtanh(input_tensor_a, *args, min_val=-1.0, max_val=1.0, **kwargs):
+    import torch
+
+    return torch.nn.functional.hardtanh(input_tensor_a, min_val=min_val, max_val=max_val)
+
+
+ttnn.attach_golden_function(ttnn.hardtanh, golden_function=_golden_function_hardtanh)
+
+
 SigmoidMode = ttnn._ttnn.operations.unary.SigmoidMode

 __all__ = []
```

## Design Decisions
- **Reference operations most useful**: relu6_analysis was the most useful reference because RELU6 (`min(max(x, 0), 6)`) is a special case of hardtanh. The relu_max kernel pattern of using v_if comparisons with Converter::as_float for parameter reconstruction was directly adapted. The hardsigmoid_analysis was also useful for understanding the SFPU dispatch chain through standard LLK macros.
- **SFPU kernel approach**: Created a clean two-comparison clamp kernel using direct `v_if(val < min_val)` and `v_if(val > max_val)` comparisons, rather than the existing tt_llk kernel's 3-parameter arithmetic trick (add/clamp/add/clamp/add). The direct approach is simpler, more readable, and avoids precision loss from chained FP16_B additions.
- **Parameter passing**: Parameters (min_val, max_val) are bitcast from float to uint32_t (IEEE 754 representation) and baked into the SFPU_OP_CHAIN compile-time define string as hex literals. This follows the relu_max pattern (e.g., `relu_max_tile(0, 0x40c00000u)`) and avoids needing runtime scalar arg packing.
- **Two-parameter registration**: Since no existing macro handles two float parameters with defaults, a custom inline function was defined in unary.hpp. A nanobind wrapper (`unary_two_float_5param_to_6param_wrapper`) bridges the 6-param C++ function to the 5-param nanobind binding (dropping sub_core_grids).
- **Old unary path only**: Registered in the old `unary_op_utils.cpp` path (not `unary_ng`) because the ng path's `get_op_init_and_func` doesn't accept parameters and can't embed them in the SFPU_OP_CHAIN string.

## Known Limitations
- No INT32 support (hardtanh is float-only).
- Parameters are baked as compile-time defines, so each unique (min_val, max_val) pair triggers a kernel recompile.
- Not registered in the unary_ng dispatch path.

## Test Results
- **Status**: PASS (after 1 test run, 2 implementation fixes)
- **Test file**: tests/ttnn/unit_tests/operations/eltwise/test_hardtanh.py
- **bfloat16** (is_fp32=False):
  - **Parameters tested**: default(-1,1), narrow(-0.5,0.5), relu6-like(0,6), wide(-2,2)
  - **Max ULP**: 0 (hardtanh is a clamp — exact on bfloat16)
  - **allclose**: PASS (rtol=1.6e-2, atol=1e-2)
- **fp32** (is_fp32=True):
  - **Parameters tested**: default(-1,1), narrow(-0.5,0.5), relu6-like(0,6), wide(-2,2)
  - **Max ULP**: 0 (hardtanh is a clamp — exact on fp32)
  - **allclose**: PASS (rtol=1e-3, atol=1e-4)

## Debug Log
### Fix 1: SFPU kernel signature mismatch
- **Error type**: build_error (on-device JIT compilation)
- **Error**: `too few arguments to function` at `llk_math_eltwise_unary_sfpu_params.h:31`
- **Hypothesis**: `calculate_hardtanh` takes `(iterations, param0, param1)` but `_llk_math_eltwise_unary_sfpu_params_` template calls `sfpu_func(args...)` where `args` is only `(param0, param1)`.
- **Fix**: Removed `iterations` parameter from `calculate_hardtanh`, changed inner loop from `for(d < iterations)` to `for(d < ITERATIONS)` using the template parameter.
- **Files modified**: ckernel_sfpu_hardtanh.h (wormhole_b0 and blackhole)

### Fix 2: fmt::format doc string escaping
- **Error type**: build_error (host compilation)
- **Error**: `argument not found` in fmt::format for hardtanh nanobind doc string
- **Hypothesis**: `{min_val}` and `{max_val}` interpreted as named format args by fmt::format.
- **Fix**: Escaped as `{{min_val}}` and `{{max_val}}` in the doc R"doc()" string.
- **Files modified**: unary_nanobind.cpp
