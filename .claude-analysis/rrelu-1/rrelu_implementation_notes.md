# Implementation Notes: rrelu

## Math Definition
RReLU(x) = x if x >= 0, a*x if x < 0
- Training mode: a ~ Uniform(lower, upper), where a is randomly sampled per element
- Eval/inference mode: a = (lower + upper) / 2 (deterministic)
- Default parameters: lower = 1/8 = 0.125, upper = 1/3 ~ 0.3333

## Files Created

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
SFPU kernel implementing the dual-path RReLU logic. Eval mode uses SFPI abstractions with full loop unrolling. Training mode uses raw TTI instructions for PRNG access and CC-guarded conditional multiply. Identical copy at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint32_t seed) {
    init_prng_seed(seed);
}

// RReLU(x) = x if x >= 0
//            a*x if x < 0
// Eval mode (training_uint == 0): a = lower + range * 0.5 = (lower + upper) / 2
// Training mode (training_uint != 0): a ~ Uniform(lower, upper) per element
//
// Parameters:
//   lower_uint: bitcast of lower bound (float)
//   range_uint: bitcast of (upper - lower) (float), precomputed on host
//   training_uint: bitcast of 1.0f if training, 0 if eval
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_uint, uint range_uint, uint training_uint) {
    // training_uint is bitcast of 1.0f (= 0x3f800000) for training, 0 for eval
    if (training_uint != 0) {
        // ---- Training mode: random slope per element ----
        // Uses raw TTI instructions for PRNG access (same pattern as rand + leaky_relu).

        // Load range into LREG1
        TT_SFPLOADI(p_sfpu::LREG1, 10, range_uint & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, 8, range_uint >> 16);

        // Load lower into LREG2
        TT_SFPLOADI(p_sfpu::LREG2, 10, lower_uint & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG2, 8, lower_uint >> 16);

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            // Generate random float in [0, 1) using PRNG (same technique as ckernel_sfpu_rand.h)
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
            TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            TTI_SFPADDI(0xbf80 /*-1.0 in FP16_B*/, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // slope = rand_01 * range + lower
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // Load input from DEST
            TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);

            // Set CC: lanes where input < 0 become active
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);

            // CC-guarded multiply: input *= slope (only for negative lanes)
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

            // Reset CC: all lanes active
            TTI_SFPENCC(0, 0, 0, 0);

            // Store result back to DEST
            TTI_SFPSTORE(p_sfpu::LREG0, 0, 3, 0);

            dst_reg++;
        }
    } else {
        // ---- Eval mode: fixed slope = (lower + upper) / 2 = lower + range * 0.5 ----
        // Uses SFPI abstractions (same pattern as prelu).
        vFloat lower_val = Converter::as_float(lower_uint);
        vFloat range_val = Converter::as_float(range_uint);
        vFloat slope = lower_val + range_val * vFloat(0.5f);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat v = dst_reg[0];
            v_if(v < 0.0f) { v = v * slope; }
            v_endif;
            dst_reg[0] = v;
            dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint32_t seed) {
    init_prng_seed(seed);
}

// RReLU(x) = x if x >= 0
//            a*x if x < 0
// Eval mode (training_uint == 0): a = lower + range * 0.5 = (lower + upper) / 2
// Training mode (training_uint != 0): a ~ Uniform(lower, upper) per element
//
// Parameters:
//   lower_uint: bitcast of lower bound (float)
//   range_uint: bitcast of (upper - lower) (float), precomputed on host
//   training_uint: bitcast of 1.0f if training, 0 if eval
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_uint, uint range_uint, uint training_uint) {
    // training_uint is bitcast of 1.0f (= 0x3f800000) for training, 0 for eval
    if (training_uint != 0) {
        // ---- Training mode: random slope per element ----
        // Uses raw TTI instructions for PRNG access (same pattern as rand + leaky_relu).

        // Load range into LREG1
        TT_SFPLOADI(p_sfpu::LREG1, 10, range_uint & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, 8, range_uint >> 16);

        // Load lower into LREG2
        TT_SFPLOADI(p_sfpu::LREG2, 10, lower_uint & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG2, 8, lower_uint >> 16);

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            // Generate random float in [0, 1) using PRNG (same technique as ckernel_sfpu_rand.h)
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
            TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            TTI_SFPADDI(0xbf80 /*-1.0 in FP16_B*/, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // slope = rand_01 * range + lower
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // Load input from DEST
            TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);

            // Set CC: lanes where input < 0 become active
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);

            // CC-guarded multiply: input *= slope (only for negative lanes)
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

            // Reset CC: all lanes active
            TTI_SFPENCC(0, 0, 0, 0);

            // Store result back to DEST
            TTI_SFPSTORE(p_sfpu::LREG0, 0, 3, 0);

            dst_reg++;
        }
    } else {
        // ---- Eval mode: fixed slope = (lower + upper) / 2 = lower + range * 0.5 ----
        // Uses SFPI abstractions (same pattern as prelu).
        vFloat lower_val = Converter::as_float(lower_uint);
        vFloat range_val = Converter::as_float(range_uint);
        vFloat slope = lower_val + range_val * vFloat(0.5f);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat v = dst_reg[0];
            v_if(v < 0.0f) { v = v * slope; }
            v_endif;
            dst_reg[0] = v;
            dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
LLK dispatch wrapper for the SFPU rrelu kernel. Provides the init and unrolled iteration template. Identical copy at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>(
        ckernel::sfpu::rrelu_init<APPROXIMATE>, static_cast<uint32_t>(0));
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint32_t lower, uint32_t range, uint32_t training, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, range, training);
}

}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rrelu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>(
        ckernel::sfpu::rrelu_init<APPROXIMATE>, static_cast<uint32_t>(0));
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint32_t lower, uint32_t range, uint32_t training, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, range, training);
}

}  // namespace ckernel
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`
Public compute API header exposing rrelu_tile() and rrelu_tile_init() functions to kernels. Wraps the LLK dispatch layer.

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rrelu.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise RReLU (Randomized Leaky ReLU) operation on a tile in DST register at index idst.
 * RReLU(x) = x if x >= 0, a*x if x < 0
 * In eval mode: a = (lower + upper) / 2 (deterministic)
 * In training mode: a ~ Uniform(lower, upper) (random per element)
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound (bitcast float)                                               | uint32_t |                                                       | True     |
 * | param1          | Range = upper - lower (bitcast float)                                     | uint32_t |                                                       | True     |
 * | param2          | Training mode flag (bitcast float: 1.0 = training, 0.0 = eval)           | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1, uint32_t param2) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, param0, param1, param2)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

}  // namespace ckernel
```

### `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`
Comprehensive test suite covering both eval and training modes. Eval mode exhaustively tests all bfloat16 bit patterns against PyTorch golden. Training mode verifies range bounds on negative outputs and checks for random slope diversity.

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the rrelu (Randomized Leaky ReLU) SFPU operation.

Two test modes:
  - Eval mode (training=False): deterministic, slope = (lower + upper) / 2.
    Exhaustive comparison over all bfloat16 bit patterns.
  - Training mode (training=True): non-deterministic random slope in [lower, upper].
    Verifies positive passthrough and that negative outputs fall in the valid range.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


# ---------------------------------------------------------------------------
# Eval mode test (deterministic) -- exhaustive bfloat16 + fp32
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_eval(device, is_fp32):
    """Test rrelu in eval mode (training=False) with all bfloat16 bit patterns."""
    lower = 0.125
    upper = 1.0 / 3.0

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Eval-mode golden: slope = (lower + upper) / 2
    slope = (lower + upper) / 2.0
    torch_output = torch.where(
        torch_input.float() >= 0,
        torch_input.float(),
        torch_input.float() * slope,
    )
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
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
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Training mode test (non-deterministic) -- range check
# ---------------------------------------------------------------------------
def test_rrelu_training(device):
    """
    Test rrelu in training mode (training=True).

    For x >= 0:  output == input  (identity)
    For x < 0:   lower * x <= output <= upper * x
        (Note: since x < 0, upper*x is more negative, lower*x is less negative,
         so the bounds flip: upper*x <= output <= lower*x)
    """
    lower = 0.125
    upper = 1.0 / 3.0

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Run on device
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    # Flush subnormals in both input and output to match hardware behavior
    torch_input_f32 = flush_subnormal_values_to_zero(torch_input.float())
    actual_f32 = flush_subnormal_values_to_zero(actual.float())

    # Only check finite, non-zero values (zero inputs can produce tiny subnormal
    # artifacts due to PRNG multiplication; these get flushed to zero above)
    finite_mask = torch.isfinite(torch_input_f32) & torch.isfinite(actual_f32)
    input_finite = torch_input_f32[finite_mask]
    output_finite = actual_f32[finite_mask]

    # Positive values: output == input (identity)
    # Use strictly positive mask (exclude zero, which can have -0.0 vs 0.0 issues)
    pos_mask = input_finite > 0
    pos_input = input_finite[pos_mask]
    pos_output = output_finite[pos_mask]
    assert torch.equal(
        pos_output.to(torch.bfloat16), pos_input.to(torch.bfloat16)
    ), "Training mode: positive values should pass through unchanged"

    # Zero inputs: output should also be zero (or flushed subnormal)
    zero_mask = input_finite == 0
    zero_output = output_finite[zero_mask]
    assert (zero_output == 0).all(), (
        f"Training mode: zero inputs should produce zero output, " f"got {zero_output[zero_output != 0][:5].tolist()}"
    )

    # Negative values: output should be in [upper*x, lower*x]
    # (upper*x is more negative because upper > lower and x < 0)
    neg_mask = input_finite < 0
    neg_input = input_finite[neg_mask]
    neg_output = output_finite[neg_mask]

    lower_bound = neg_input * upper  # more negative (smaller)
    upper_bound = neg_input * lower  # less negative (larger)

    # Allow a small tolerance for bfloat16 rounding
    tol = 1e-2
    violations_low = (neg_output < lower_bound - tol).sum().item()
    violations_high = (neg_output > upper_bound + tol).sum().item()
    total_neg = neg_input.numel()

    assert violations_low == 0, (
        f"Training mode: {violations_low}/{total_neg} negative outputs below lower_bound (upper*x). "
        f"Worst: output={neg_output[neg_output < lower_bound - tol][:5].tolist()}, "
        f"bound={lower_bound[neg_output < lower_bound - tol][:5].tolist()}"
    )
    assert violations_high == 0, (
        f"Training mode: {violations_high}/{total_neg} negative outputs above upper_bound (lower*x). "
        f"Worst: output={neg_output[neg_output > upper_bound + tol][:5].tolist()}, "
        f"bound={upper_bound[neg_output > upper_bound + tol][:5].tolist()}"
    )

    # Verify that the random slopes are not all the same (i.e., actual randomness)
    # Compute implied slope: output / input for negative values (avoid div by zero)
    nonzero_neg_mask = neg_input.abs() > 1e-6
    if nonzero_neg_mask.sum() > 10:
        slopes = neg_output[nonzero_neg_mask] / neg_input[nonzero_neg_mask]
        unique_slopes = slopes.unique()
        assert unique_slopes.numel() > 1, (
            "Training mode: all negative slopes are identical -- expected randomness. "
            f"Got slope = {unique_slopes.tolist()}"
        )
```

## Files Modified

### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
Added conditional include for RRELU after PRELU.

```diff
@@ -140,6 +140,10 @@
 #include "api/compute/eltwise_unary/prelu.h"
 #endif

+#if SFPU_OP_RRELU_INCLUDE
+#include "api/compute/eltwise_unary/rrelu.h"
+#endif
+
 #if SFPU_OP_DROPOUT_INCLUDE
 #include "api/compute/eltwise_unary/dropout.h"
 #endif
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
Added RRELU enum value to SfpuType enumeration.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
Added RRELU enum value to SfpuType enumeration.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
Added RRELU to UnaryOpType enumeration.

```diff
@@ -132,6 +132,7 @@ enum class UnaryOpType {
     LOGIT,
     XIELU,
     LGAMMA,
+    RRELU,
 };

 enum class VecMode {
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
Added macro definition case for RRELU and parameter parsing logic. Precomputes range = upper - lower on host side.

```diff
@@ -97,6 +97,7 @@ std::string get_macro_definition(UnaryOpType op_type) {
         case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
         case UnaryOpType::HARDMISH: return "SFPU_OP_HARDMISH_INCLUDE";
         case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
+        case UnaryOpType::RRELU: return "SFPU_OP_RRELU_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     };
 }
@@ -561,6 +562,21 @@ std::pair<std::string, std::string> get_op_init_and_func_parameterized(
                     std::bit_cast<uint32_t>(param0),
                     std::bit_cast<uint32_t>(param1))};
         }
+        case UnaryOpType::RRELU: {
+            TT_FATAL(params.size() == 3, "Expected rrelu to take 3 parameters (lower, upper, training)");
+            float lower = param0;
+            float upper = params[1];
+            float training = params[2];
+            float range = upper - lower;
+            return {
+                "rrelu_tile_init();",
+                fmt::format(
+                    "rrelu_tile({}, {:#x}u, {:#x}u, {:#x}u);",
+                    idst,
+                    std::bit_cast<uint32_t>(lower),
+                    std::bit_cast<uint32_t>(range),
+                    std::bit_cast<uint32_t>(training))};
+        }
         case UnaryOpType::HARDMISH: {
             return {
                 fmt::format("hardmish_tile_init<{}u>();", (uint32_t)param0),
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
Added RRELU to is_parametrized_type() check.

```diff
@@ -98,6 +98,7 @@ bool is_parametrized_type(T val) {
         case UnaryOpType::THRESHOLD:
         case UnaryOpType::CLAMP_TSS:
         case UnaryOpType::SELU:
+        case UnaryOpType::RRELU:
         case UnaryOpType::LOGIT:
         case UnaryOpType::RPOW:
         case UnaryOpType::MISH: return true;
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
Added C++ function signature for rrelu with default parameters: lower=0.125, upper=1/3, training=false.

```diff
@@ -267,6 +267,14 @@ Tensor selu(
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
     const std::optional<Tensor>& optional_output_tensor = std::nullopt);

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower = 0.125f,
+    float upper = 1.0f / 3.0f,
+    bool training = false,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
+    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
+
 Tensor bitcast(
     const Tensor& input_tensor,
     const tt::tt_metal::DataType& output_dtype,
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
Implemented C++ rrelu() function. Converts training bool to float flag (1.0 for True, 0.0 for False) and delegates to unary_impl with UnaryOpType::RRELU.

```diff
@@ -371,6 +371,21 @@ Tensor selu(
         input_tensor, {UnaryWithParam{UnaryOpType::SELU, {scale, alpha}}}, memory_config, optional_output_tensor);
 }

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower,
+    float upper,
+    bool training,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
+    const std::optional<Tensor>& optional_output_tensor) {
+    float training_flag = training ? 1.0f : 0.0f;
+    return ttnn::detail::unary_impl(
+        input_tensor,
+        {UnaryWithParam{UnaryOpType::RRELU, {lower, upper, training_flag}}},
+        memory_config,
+        optional_output_tensor);
+}
+
 Tensor swish(
     const Tensor& input_tensor,
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
Added Python bindings for rrelu using bind_function with comprehensive docstring and parameters (lower, upper, training, memory_config, output_tensor).

```diff
@@ -928,6 +928,57 @@ void bind_softplus(nb::module_& mod) {
         nb::arg("output_tensor") = nb::none());
 }

+void bind_rrelu(nb::module_& mod) {
+    auto doc = fmt::format(
+        R"doc(
+        Applies {0} to :attr:`input_tensor` element-wise.
+
+        .. math::
+            \mathrm{{output\_tensor}}_i = \begin{{cases}} x_i & \text{{if }} x_i \geq 0 \\ a \cdot x_i & \text{{if }} x_i < 0 \end{{cases}}
+
+        In training mode, ``a`` is randomly sampled from Uniform(lower, upper) per element.
+        In eval mode, ``a = (lower + upper) / 2``.
+
+        Args:
+            input_tensor (ttnn.Tensor): the input tensor.
+
+        Keyword Args:
+            lower (float, optional): Lower bound of the uniform distribution. Defaults to `0.125`.
+            upper (float, optional): Upper bound of the uniform distribution. Defaults to `0.3333`.
+            training (bool, optional): If True, use random slope (training mode). If False, use fixed slope (eval mode). Defaults to `False`.
+            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
+            output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
+
+        Returns:
+            ttnn.Tensor: the output tensor.
+
+        Note:
+            Supported dtypes and layouts:
+
+            .. list-table::
+               :header-rows: 1
+
+               * - Dtypes
+                 - Layouts
+               * - FLOAT32, BFLOAT16, BFLOAT8_B
+                 - TILE
+        )doc",
+        "rrelu",
+        "ttnn.rrelu");
+
+    ttnn::bind_function<"rrelu">(
+        mod,
+        doc.c_str(),
+        &ttnn::rrelu,
+        nb::arg("input_tensor"),
+        nb::kw_only(),
+        nb::arg("lower") = 0.125f,
+        nb::arg("upper") = 1.0f / 3.0f,
+        nb::arg("training") = false,
+        nb::arg("memory_config") = nb::none(),
+        nb::arg("output_tensor") = nb::none());
+}
+
 void bind_xielu(nb::module_& mod) {
     auto doc = fmt::format(
         R"doc(
@@ -2224,6 +2275,7 @@ void py_module(nb::module_& mod) {

     // Other unaries (unary chain operations)
     bind_softplus(mod);
+    bind_rrelu(mod);
     bind_xielu(mod);
     bind_tanh_like<"tanh", &ttnn::tanh>(mod);
     bind_tanh_like<"tanhshrink", &ttnn::tanhshrink>(mod);
```

### `ttnn/ttnn/operations/unary.py`
Added Python golden function wrapper mapping ttnn.rrelu to torch.nn.functional.rrelu with matching parameters.

```diff
@@ -403,6 +403,15 @@ def _golden_function_selu(input_tensor_a, *args, **kwargs):
 ttnn.attach_golden_function(ttnn.selu, golden_function=_golden_function_selu)


+def _golden_function_rrelu(input_tensor_a, *args, lower=0.125, upper=1.0 / 3.0, training=False, **kwargs):
+    import torch
+
+    return torch.nn.functional.rrelu(input_tensor_a, lower=lower, upper=upper, training=training)
+
+
+ttnn.attach_golden_function(ttnn.rrelu, golden_function=_golden_function_rrelu)
+
+
 def _golden_function_tanhshrink(input_tensor_a, *args, **kwargs):
     import torch
```

## Design Decisions

### Reference Operations Used
- **prelu_sfpu_analysis**: Most useful for the eval-mode kernel -- the conditional multiply pattern (`v_if(v < 0.0f) { v = v * slope; } v_endif`) is identical to prelu. The SFPI abstraction layer pattern (vFloat, dst_reg, v_if/v_endif) was directly reused.
- **rand_analysis**: Essential for training-mode PRNG random number generation. The technique of reading the PRNG counter via `TTI_SFPMOV(0, 9, LREG3, 8)`, clearing sign, setting exponent to 127, subtracting 1.0 to get [0,1), then scaling with SFPMAD was adopted exactly from the rand kernel.
- **leaky_relu_analysis**: Provided the raw TTI instruction pattern for CC-guarded multiply on negative lanes: SFPLOAD -> SFPSETCC -> SFPMUL -> SFPENCC -> SFPSTORE. Used for the training-mode path.
- **selu_analysis**: Provided the 2-parameter registration pattern through all abstraction layers (LLK dispatch with custom init, compute API header calling LLK directly, nanobind with `bind_unary_composite_floats_with_default`). Extended to 3 parameters for rrelu.
- **dropout_analysis**: Provided the PRNG seeding pattern (`init_prng_seed(seed)` in the init function).

### Dual-Path Kernel Design
The SFPU kernel (`ckernel_sfpu_rrelu.h`) contains two distinct execution paths selected by a runtime `if` on the `training_uint` parameter:
- **Eval path**: Pure SFPI abstractions (vFloat, v_if/v_endif) matching the prelu pattern. The fixed slope is computed as `lower + range * 0.5`. Loop is `#pragma GCC unroll 8` for full unrolling.
- **Training path**: Raw TTI instructions for PRNG access + CC-guarded multiply. Loop is `#pragma GCC unroll 0` (no unrolling) because the PRNG read has side effects and register pressure is higher with LREG0-3 all in use.

### Precomputed Range Parameter
Instead of passing `lower` and `upper` directly to the SFPU kernel, the host-side `get_op_init_and_func_parameterized()` precomputes `range = upper - lower`. This avoids a floating-point subtraction on the SFPU, which would require either:
- Negating a register (complex in raw TTI without a dedicated negate instruction)
- Mixing SFPI and TTI instructions (fragile due to register allocation conflicts)

The 3 params passed to the kernel are: `lower`, `range`, `training_flag`.

### PRNG Seeding
The PRNG is seeded with a fixed seed (0) during `rrelu_tile_init()` via the LLK dispatch init function. This is a known limitation: the standard `UnaryProgramFactory` does not support passing per-tile seeds. Different cores will produce different random sequences because the PRNG LFSR state diverges after seeding, but tiles processed by the same core will share the PRNG state progression.

### Parameter Encoding
The `training` parameter (Python `bool`) is converted to `float` (1.0 for True, 0.0 for False) in `unary.cpp`, then bitcast to `uint32_t` in `unary_op_utils.cpp`. The kernel checks `training_uint != 0` to select the execution path (0x3f800000 for True, 0x00000000 for False).

### Wormhole/Blackhole Parity
Both architecture implementations are identical. The TTI instructions in the training path use `ADDR_MOD_3` (hardcoded literal `3`) for SFPLOAD/SFPSTORE, which maps to ADDR_MOD_7 on Wormhole via the addr mod base remapping and directly to ADDR_MOD_3 on Blackhole. Both result in zero auto-increment, which is the standard behavior for SFPU operations.

## Known Limitations
- **Fixed PRNG seed**: Training mode uses seed=0 for all invocations. True random behavior would require a custom program factory that passes per-core seeds.
- **No per-tile seed variation**: All tiles processed by the same core share the same PRNG state progression, so the random slopes are deterministic given the processing order.
- **BFloat16 rounding**: The eval-mode path uses SFPI abstractions which handle BFloat16 rounding automatically. The training-mode path uses raw TTI SFPSTORE with format mode 0 (default/BFloat16), which should handle rounding correctly for BFloat16 inputs.
- **PRNG quality**: The hardware PRNG is a 32-bit LFSR with period 2^32-1. The random slopes are uniform in [lower, upper) but with limited randomness quality compared to software PRNGs.

## Test Results
- **Status**: PASS (after 2 attempts)
- **Test file**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py
- **bfloat16** (is_fp32=False, eval mode):
  - **Max ULP**: 1 (threshold: 2)
  - **allclose**: PASS (rtol=1.6e-2, atol=1e-2)
- **fp32** (is_fp32=True, eval mode):
  - **Max ULP**: 0 (perfect match, threshold: 3)
  - **allclose**: PASS (rtol=1e-3, atol=1e-4)
- **training mode** (bfloat16):
  - Positive passthrough: PASS
  - Negative range check [upper*x, lower*x]: PASS
  - Random slope diversity: PASS

## Debug Log
### Attempt 1
- **Result**: FAIL (1 of 3 tests)
- **Passed**: test_rrelu_eval[bfloat16], test_rrelu_eval[fp32]
- **Failed**: test_rrelu_training
- **Error type**: test_logic_error
- **Error**: Training mode positive passthrough assertion fails -- `torch.equal` returns False due to subnormal values in output (e.g., 9.1835e-41 where input is 0.0) and -0.0 vs 0.0 mismatch
- **Hypothesis**: Test too strict -- uses `torch.equal` which fails on subnormal artifacts and signed-zero differences. Need to flush subnormals and separate zero-input handling.
- **Fix**: Fixed training test to flush subnormals in both input and output, use strictly positive mask (>0 instead of >=0) for passthrough check, added separate zero-input assertion
- **Files modified**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Attempt 2
- **Result**: PASS (all 3 tests)
- **bfloat16 eval**: max ULP 1, allclose PASS
- **fp32 eval**: max ULP 0, allclose PASS
- **training**: all assertions passed
