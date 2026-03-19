# Implementation Notes: smelu

## Math Definition
SmeLU(x, beta) = x if x >= beta; (x + beta)^2 / (4*beta) if |x| <= beta; 0 if x < -beta

Where beta is a positive parameter (default = 2.0). This is a smooth approximation of ReLU with a quadratic transition region.

## Files Created

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_smelu.h`
SFPU compute kernel implementing the SmeLU activation function. Computes the three regions of the piecewise function using conditional execution on the SFPU.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_smelu(uint32_t param0, uint32_t param1) {
    // SmeLU(x, beta) = x if x >= beta; (x + beta)^2 / (4*beta) if |x| <= beta; 0 if x < -beta
    // param0 = beta (bit-cast float as uint32)
    // param1 = 1/(4*beta) (bit-cast float as uint32, precomputed on host)
    sfpi::vFloat beta = Converter::as_float(param0);
    sfpi::vFloat inv_4beta = Converter::as_float(param1);
    sfpi::vFloat neg_beta = -beta;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // Default: middle region (|x| <= beta): (x + beta)^2 * inv_4beta
        sfpi::vFloat x_plus_beta = v + beta;
        sfpi::dst_reg[0] = x_plus_beta * x_plus_beta * inv_4beta;
        v_if(v >= beta) {
            // x >= beta: identity
            sfpi::dst_reg[0] = v;
        }
        v_elseif(v < neg_beta) {
            // x < -beta: zero
            sfpi::dst_reg[0] = sfpi::vConst0;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_smelu.h`
Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_smelu.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_smelu(uint32_t param0, uint32_t param1) {
    // SmeLU(x, beta) = x if x >= beta; (x + beta)^2 / (4*beta) if |x| <= beta; 0 if x < -beta
    // param0 = beta (bit-cast float as uint32)
    // param1 = 1/(4*beta) (bit-cast float as uint32, precomputed on host)
    sfpi::vFloat beta = Converter::as_float(param0);
    sfpi::vFloat inv_4beta = Converter::as_float(param1);
    sfpi::vFloat neg_beta = -beta;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // Default: middle region (|x| <= beta): (x + beta)^2 * inv_4beta
        sfpi::vFloat x_plus_beta = v + beta;
        sfpi::dst_reg[0] = x_plus_beta * x_plus_beta * inv_4beta;
        v_if(v >= beta) {
            // x >= beta: identity
            sfpi::dst_reg[0] = v;
        }
        v_elseif(v < neg_beta) {
            // x < -beta: zero
            sfpi::dst_reg[0] = sfpi::vConst0;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

### `tests/ttnn/unit_tests/operations/eltwise/test_smelu.py`
Unit test suite for the SmeLU operation with parametrized beta values and tensor shapes.

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    device = ttnn.CreateDevice(device_id=0)
    ttnn.SetDefaultDevice(device)
    yield device
    ttnn.close_device(device)


def assert_with_pcc(expected, actual, pcc_threshold=0.999):
    expected = expected.float().flatten()
    actual = actual.float().flatten()
    if torch.all(expected == 0) and torch.all(actual == 0):
        pcc = 1.0
    else:
        pcc = torch.corrcoef(torch.stack([expected, actual]))[0, 1].item()
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} < {pcc_threshold}"
    return pcc


def torch_smelu(x, beta):
    return torch.where(
        x >= beta, x, torch.where(x <= -beta, torch.zeros_like(x), ((x + beta) ** 2) / (4.0 * beta))
    )


@pytest.mark.parametrize("h, w", [(32, 32), (64, 64), (128, 128)])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_smelu(device, h, w, beta):
    torch.manual_seed(0)
    torch_input = torch.randn((h, w), dtype=torch.bfloat16)
    torch_output = torch_smelu(torch_input.to(torch.float32), beta).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.smelu(tt_input, beta=beta)
    tt_output = ttnn.to_torch(tt_output)

    pcc = assert_with_pcc(torch_output, tt_output, 0.999)
    print(f"PCC for shape=({h},{w}), beta={beta}: {pcc:.6f}")
```

## Files Modified

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_activations.h`
Added SmeLU kernel registration and dispatch functions. Adds include for the new SmeLU kernel and wraps the SFPU calculation in LLK template functions for both init and compute stages.

```diff
@@ -9,6 +9,7 @@
 #include "ckernel_sfpu_softsign.h"
 #include "ckernel_sfpu_softshrink.h"
 #include "ckernel_sfpu_celu.h"
+#include "ckernel_sfpu_smelu.h"

 namespace ckernel {

@@ -69,4 +70,17 @@ inline void llk_math_eltwise_unary_sfpu_softshrink(uint dst_index, uint param0,
         ckernel::sfpu::calculate_softshrink<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
 }

+// smelu
+template <bool APPROXIMATE>
+inline void llk_math_eltwise_unary_sfpu_smelu_init() {
+    llk_math_eltwise_unary_sfpu_init<SfpuType::smelu, APPROXIMATE>();
+}
+
+template <bool APPROXIMATE, int ITERATIONS = 8>
+inline void llk_math_eltwise_unary_sfpu_smelu(
+    uint dst_index, uint32_t param0, uint32_t param1, int vector_mode = (int)VectorMode::RC) {
+    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
+        ckernel::sfpu::calculate_smelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1);
+}
+
 }  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_activations.h`
Added SmeLU kernel registration and dispatch functions. Adds include for the new SmeLU kernel and wraps the SFPU calculation in LLK template functions for both init and compute stages.

```diff
@@ -9,10 +9,10 @@
 #include "ckernel_sfpu_softsign.h"
 #include "ckernel_sfpu_softshrink.h"
 #include "ckernel_sfpu_celu.h"
+#include "ckernel_sfpu_smelu.h"

 namespace ckernel {

-// Hardsigmoid
 template <bool APPROXIMATE>
 inline void llk_math_eltwise_unary_sfpu_hardsigmoid_init() {
     llk_math_eltwise_unary_sfpu_init<SfpuType::hardsigmoid, APPROXIMATE>(
@@ -70,4 +70,17 @@ inline void llk_math_eltwise_unary_sfpu_softshrink(uint dst_index, uint param0,
         ckernel::sfpu::calculate_softshrink<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0);
 }

+// smelu
+template <bool APPROXIMATE>
+inline void llk_math_eltwise_unary_sfpu_smelu_init() {
+    llk_math_eltwise_unary_sfpu_init<SfpuType::smelu, APPROXIMATE>();
+}
+
+template <bool APPROXIMATE, int ITERATIONS = 8>
+inline void llk_math_eltwise_unary_sfpu_smelu(
+    uint dst_index, uint32_t param0, uint32_t param1, int vector_mode = (int)VectorMode::RC) {
+    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
+        ckernel::sfpu::calculate_smelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, param0, param1);
+}
+
 }  // namespace ckernel
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/activations.h`
Added SmeLU public compute API for the tile-level compute interface.

```diff
@@ -100,4 +100,28 @@ ALWI void celu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_celu_init<APPROX>
  */
 ALWI void softshrink_tile_init() { MATH((llk_math_eltwise_unary_sfpu_softshrink_init<APPROX>())); }

+// clang-format off
+/**
+* Performs element-wise smelu operation. The DST
+* register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
+* compute engine.
+*
+* Return value: None
+*
+* | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
+* |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
+* | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
+* | param0          | The beta parameter (bit-cast float as uint32)                              | uint32_t |                                                       | True     |
+* | param1          | The 1/(4*beta) parameter (bit-cast float as uint32)                        | uint32_t |                                                       | True     |
+*/
+// clang-format on
+ALWI void smelu_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
+    MATH((llk_math_eltwise_unary_sfpu_smelu<APPROX>(idst, param0, param1)));
+}
+
+/**
+ * Please refer to documentation for any_init.
+ */
+ALWI void smelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_smelu_init<APPROX>())); }
+
 }  // namespace ckernel
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
Added SmeLU operation type to the SfpuType enum.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    smelu,
 };
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
Added SmeLU operation type to the SfpuType enum.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    smelu,
 };
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
Added SMELU to the UnaryOpType enum.

```diff
@@ -132,6 +132,7 @@ enum class UnaryOpType {
     LOGIT,
     XIELU,
     LGAMMA,
+    SMELU,
 };

 enum class VecMode {
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
Added macro definition mapping and host-side parameter precomputation for SmeLU. Maps SmeLU to the SFPU_OP_ACTIVATIONS_INCLUDE macro family and generates code to precompute and pass both beta and 1/(4*beta) to the device kernel.

```diff
@@ -91,7 +91,8 @@ std::string get_macro_definition(UnaryOpType op_type) {
         case UnaryOpType::SOFTSHRINK:
         case UnaryOpType::SOFTSIGN:
         case UnaryOpType::HARDSIGMOID:
-        case UnaryOpType::CELU: return "SFPU_OP_ACTIVATIONS_INCLUDE";
+        case UnaryOpType::CELU:
+        case UnaryOpType::SMELU: return "SFPU_OP_ACTIVATIONS_INCLUDE";
         case UnaryOpType::THRESHOLD: return "SFPU_OP_THRESHOLD_INCLUDE";
         case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
         case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
@@ -500,6 +501,14 @@ std::pair<std::string, std::string> get_op_init_and_func_parameterized(
                     std::bit_cast<uint32_t>(1.0f / param0))};
         case UnaryOpType::HARDSHRINK:
         case UnaryOpType::LOGIT: return {};
+        case UnaryOpType::SMELU:
+            return {
+                "smelu_tile_init();",
+                fmt::format(
+                    "smelu_tile({}, {:#x}u, {:#x}u);",
+                    idst,
+                    std::bit_cast<uint32_t>(param0),
+                    std::bit_cast<uint32_t>(1.0f / (4.0f * param0)))};
         case UnaryOpType::SOFTSHRINK:
             return {
                 "softshrink_tile_init();",
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
Added SmeLU to the is_parametrized_type function to mark it as a parameterized operation.

```diff
@@ -100,7 +100,8 @@ bool is_parametrized_type(T val) {
         case UnaryOpType::SELU:
         case UnaryOpType::LOGIT:
         case UnaryOpType::RPOW:
-        case UnaryOpType::MISH: return true;
+        case UnaryOpType::MISH:
+        case UnaryOpType::SMELU: return true;
         default: return false;
     }
     return false;
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
Registered the SmeLU C++ operation using the REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER macro.

```diff
@@ -189,6 +189,7 @@ REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_min, RELU_MIN)
 REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_remainder, REMAINDER)
 REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(celu, CELU)
 REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW)
+REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(smelu, SMELU)

 UNARY_OP_SCALAR_VARIANT(fill, FILL)
 UNARY_OP_SCALAR_VARIANT(power, POWER)
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
Added Python nanobind binding for SmeLU with default beta=2.0.

```diff
@@ -2091,6 +2091,8 @@ void py_module(nb::module_& mod) {
         mod, "exponent", "exponent value. Non-positive values are not supported.", "");
     bind_unary_operation_with_float_parameter_default<"celu", &ttnn::celu>(
         mod, "alpha", "The alpha parameter for the CELU function", 1.0f, "", R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
+    bind_unary_operation_with_float_parameter_default<"smelu", &ttnn::smelu>(
+        mod, "beta", "The beta parameter for the SmeLU function", 2.0f, "", R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");

     bind_unary_operation_with_scalar_parameter<"fill", &ttnn::fill>(
         mod,
```

### `ttnn/ttnn/operations/unary.py`
Added golden function for SmeLU and attached it to the operation for validation testing.

```diff
@@ -600,6 +600,21 @@ def _golden_function_celu(input_tensor_a, *args, alpha=1.0, **kwargs):
 ttnn.attach_golden_function(ttnn.celu, golden_function=_golden_function_celu)


+def _golden_function_smelu(input_tensor_a, *args, beta=2.0, **kwargs):
+    import torch
+
+    x = input_tensor_a.to(torch.float32)
+    result = torch.where(
+        x >= beta,
+        x,
+        torch.where(x < -beta, torch.zeros_like(x), (x + beta) ** 2 / (4.0 * beta)),
+    )
+    return result.to(input_tensor_a.dtype)
+
+
+ttnn.attach_golden_function(ttnn.smelu, golden_function=_golden_function_smelu)
+
+
 def torch_reglu(input_tensor, *args, **kwargs):
     import torch
```

## Design Decisions

- **softshrink was the primary reference** because SmeLU is also a 3-region piecewise function with a single float parameter. The kernel structure (v_if/v_elseif/v_endif with float comparisons) directly mirrors softshrink.
- **celu was the secondary reference** for the pattern of passing two precomputed parameters (alpha and alpha_recip) to avoid device-side division. SmeLU passes beta and 1/(4*beta).
- **Added to the activations family** (SFPU_OP_ACTIVATIONS_INCLUDE) rather than creating a new include guard, since softshrink, celu, hardsigmoid, and softsign already share this family. This avoids adding a new entry to sfpu_split_includes.h.
- **SfpuType::smelu added** to llk_sfpu_types.h on both architectures since we created a new LLK dispatch entry.
- **SFPU kernel design**: The default computation path computes the quadratic middle region for all elements, then overwrites with identity (x >= beta) or zero (x < -beta) via v_if/v_elseif. This avoids needing a v_else branch and keeps the conditional logic simple.
- **Parameter precomputation**: 1/(4*beta) is computed on the host in `get_op_init_and_func_parameterized()` and passed as a second uint32 parameter, avoiding expensive division on the SFPU.
- **No bfloat16 rounding needed**: The kernel writes directly to dst_reg[0] without explicit float_to_fp16b conversion, matching the softshrink pattern. The pack stage handles format conversion.

## Known Limitations
- beta must be > 0; passing beta = 0 will cause division by zero in the host-side 1/(4*beta) precomputation.
- Precision may be limited for very small beta values where 1/(4*beta) becomes very large.
- The quadratic computation (x+beta)^2 * inv_4beta may lose precision for large |x| values near the boundary, though this is inherent to bfloat16 arithmetic.

## Debug Log

### Test Run 1 - Environment Issues (not smelu-related)
- **Issue**: Root `conftest.py` fails to import due to `ModuleNotFoundError: No module named 'ttnn.device'`. The `device` fixture is defined in root conftest which cannot load.
- **Fix**: Made test self-contained with its own `device` fixture and inline `assert_with_pcc` function. Used `--noconftest` flag. Required setting `PYTHONPATH=/localdev/vignjatijevic/tt-metal/ttnn:/localdev/vignjatijevic/tt-metal:/localdev/vignjatijevic/tt-metal/tools:/home/vignjatijevic/.local/lib/python3.10/site-packages` for ttnn to load properly.
- **Issue**: `ttnn.smelu(tt_input, beta)` failed with TypeError because `beta` is a keyword-only argument in the nanobind binding (`bind_unary_operation_with_float_parameter_default`).
- **Fix**: Changed to `ttnn.smelu(tt_input, beta=beta)`.

### Test Run 2 - JIT Kernel Compilation Failure (pre-existing environment issue)
- **Issue**: `fatal error: tensor_shape.h: No such file or directory`. The file is expected from the `tt_llk` submodule at `common/tensor_shape.h`, but the submodule was at an older version (`f9909668`) that predates the addition of this file.
- **Issue**: After fixing `tensor_shape.h`, a second error: `_sfpu_exp_fp32_accurate_` was not declared in `ckernel_sfpu_binary_pow.h`. This function was added to the tt_llk submodule in a later commit but the submodule pointer was stale.
- **Fix**: Checked out the tt_llk submodule to commit `59ea0128` which contains both `tensor_shape.h` and `_sfpu_exp_fp32_accurate_`. Cleared JIT cache.
- **Note**: These are pre-existing environment issues unrelated to the smelu implementation. The source code references symbols from a newer tt_llk submodule version than what was checked out.

### Test Run 3 - All Tests Pass
- **Result**: 9/9 tests PASSED
- **PCC values achieved** (all above 0.999 threshold):
  - beta=0.5: (32,32)=1.000000, (64,64)=1.000000, (128,128)=1.000000
  - beta=1.0: (32,32)=0.999998, (64,64)=0.999997, (128,128)=0.999999
  - beta=2.0: (32,32)=0.999990, (64,64)=0.999991, (128,128)=0.999992
- **Observation**: Smaller beta values yield higher PCC (closer to 1.0). This is expected since the quadratic transition region is narrower with smaller beta, meaning more elements fall in the identity (x >= beta) or zero (x < -beta) regions which are computed exactly.

## Test Results

All 9 parametrized test cases passed with PCC values above 0.999:
- beta=0.5: Perfect or near-perfect PCC (1.0)
- beta=1.0: PCC 0.999997-0.999998
- beta=2.0: PCC 0.999990-0.999992

The kernel implementation correctly computes the SmeLU function across multiple tensor shapes and beta values.
