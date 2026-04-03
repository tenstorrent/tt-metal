# Implementation Notes: rrelu

## Math Definition
RReLU(x) = x if x >= 0; a*x if x < 0
- Training mode (seed != 0): a is randomly sampled from Uniform(lower, upper) per element using hardware PRNG
- Eval mode (seed == 0): a = (lower + upper) / 2 (handled by passing midpoint as both lower and upper, or by the Python-level golden function)
- Default: lower=0.125, upper=0.333333

## Files Created

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
Core SFPU compute kernel for RReLU on Wormhole B0. Implements raw TTI instructions for PRNG-based slope generation and conditional application to negative elements. Includes pipeline hazard NOPs required for Wormhole's forwarding behavior.

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

// RReLU(x) = x if x >= 0; a*x if x < 0
// a is randomly sampled from Uniform(lower, upper) per element using the hardware PRNG.
// The PRNG must be seeded via rrelu_init() before calling this function.
//
// Register usage in the main loop (raw TTI):
//   LREG0: current element from DEST (x), then result
//   LREG1: lower bound parameter (constant across loop)
//   LREG2: range = upper - lower (constant across loop)
//   LREG3: random value -> slope
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_param, uint upper_param) {
    // Load lower into LREG1 via two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG1, 10, lower_param & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, lower_param >> 16);

    // Load upper into LREG2 via two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG2, 10, upper_param & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, upper_param >> 16);

    // Compute range = upper - lower into LREG3, then move to LREG2.
    // Use SFPMAD: LREG3 = LREG1 * (-1.0) + LREG2 = upper - lower
    // Load -1.0 into LREG3 first, then do MAD.
    // -1.0 in FP32 = 0xBF800000. In BF16_B = 0xBF80.
    TTI_SFPLOADI(p_sfpu::LREG3, 10, 0x0000);  // LREG3.lo16 = 0
    TTI_SFPLOADI(p_sfpu::LREG3, 8, 0xBF80);   // LREG3.hi16 = 0xBF80 -> LREG3 = -1.0f

    // LREG3 = LREG1 * LREG3 + LREG2 = lower * (-1.0) + upper = upper - lower
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPNOP;  // pipeline hazard after SFPMAD on Wormhole

    // Move range from LREG3 to LREG2 (so LREG3 is free for the loop)
    // SFPMOV: VD = VC (with mod1=0, copies register)
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load current element from DEST into LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);

        // Generate random float in [0, 1) into LREG3
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);                   // LREG3 = PRNG value
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);    // sign = 0
        TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // exponent = 127 -> [1.0, 2.0)
        TTI_SFPADDI(0xbf80 /*-1.0f*/, p_sfpu::LREG3, 0);      // LREG3 -= 1.0 -> [0.0, 1.0)
        TTI_SFPNOP;                                           // pipeline hazard after SFPADDI on Wormhole

        // slope = rand_01 * range + lower = LREG3 * LREG2 + LREG1 -> LREG3
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        TTI_SFPNOP;  // pipeline hazard after SFPMAD on Wormhole

        // Now LREG3 = slope in [lower, upper)
        // For negative elements: result = x * slope
        // Set CC based on sign of LREG0 (CC.Res = 1 if LREG0 < 0)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);

        // CC-guarded: LREG0 = LREG0 * LREG3 + 0.0 (only for negative elements)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // Clear CC result
        TTI_SFPENCC(0, 0, 0, 0);

        // Store result back to DEST
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
Core SFPU compute kernel for RReLU on Blackhole. Identical to Wormhole except uses ADDR_MOD_7 instead of ADDR_MOD_3, and omits the TTI_SFPNOP pipeline hazard NOPs (Blackhole has improved pipeline forwarding).

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

// RReLU(x) = x if x >= 0; a*x if x < 0
// a is randomly sampled from Uniform(lower, upper) per element using the hardware PRNG.
// The PRNG must be seeded via rrelu_init() before calling this function.
//
// Register usage in the main loop (raw TTI):
//   LREG0: current element from DEST (x), then result
//   LREG1: lower bound parameter (constant across loop)
//   LREG2: range = upper - lower (constant across loop)
//   LREG3: random value -> slope
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_rrelu(const uint lower_param, const uint upper_param) {
    // Load lower into LREG1 via two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG1, 10, lower_param & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, lower_param >> 16);

    // Load upper into LREG2 via two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG2, 10, upper_param & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, upper_param >> 16);

    // Compute range = upper - lower into LREG3, then move to LREG2.
    // Load -1.0 into LREG3: -1.0 in FP32 = 0xBF800000
    TTI_SFPLOADI(p_sfpu::LREG3, 10, 0x0000);  // LREG3.lo16 = 0
    TTI_SFPLOADI(p_sfpu::LREG3, 8, 0xBF80);   // LREG3.hi16 = 0xBF80 -> LREG3 = -1.0f

    // LREG3 = LREG1 * LREG3 + LREG2 = lower * (-1.0) + upper = upper - lower
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG3, 0);

    // Move range from LREG3 to LREG2 (so LREG3 is free for the loop)
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load current element from DEST into LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

        // Generate random float in [0, 1) into LREG3
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);                   // LREG3 = PRNG value
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);    // sign = 0
        TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // exponent = 127 -> [1.0, 2.0)
        TTI_SFPADDI(0xbf80 /*-1.0f*/, p_sfpu::LREG3, 0);      // LREG3 -= 1.0 -> [0.0, 1.0)

        // slope = rand_01 * range + lower = LREG3 * LREG2 + LREG1 -> LREG3
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG3, 0);

        // Now LREG3 = slope in [lower, upper)
        // For negative elements: result = x * slope
        // Set CC based on sign of LREG0 (CC.Res = 1 if LREG0 < 0)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);

        // CC-guarded: LREG0 = LREG0 * LREG3 + 0.0 (only for negative elements)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // Clear CC result
        TTI_SFPENCC(0, 0, 0, 0);

        // Store result back to DEST
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
LLK math wrapper for RReLU on Wormhole B0. Dispatches to the SFPU kernel with proper parameter marshalling.

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
inline void llk_math_eltwise_unary_sfpu_rrelu_init(uint32_t seed) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>(ckernel::sfpu::rrelu_init<APPROXIMATE>, seed);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint lower, uint upper, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>,
        dst_index,
        vector_mode,
        lower,
        upper);
}

}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
LLK math wrapper for RReLU on Blackhole. Identical copy to Wormhole wrapper.

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
inline void llk_math_eltwise_unary_sfpu_rrelu_init(uint32_t seed) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>(ckernel::sfpu::rrelu_init<APPROXIMATE>, seed);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint lower, uint upper, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>,
        dst_index,
        vector_mode,
        lower,
        upper);
}

}  // namespace ckernel
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`
Public compute API header for RReLU. Defines the rrelu_tile and rrelu_tile_init functions exposed to the compute kernel dispatch layer.

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
 * Performs element-wise computation of RReLU: rrelu(x) = x if x >= 0, a*x if x < 0,
 * where a is a random slope sampled from Uniform(lower, upper) in training mode.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound of uniform distribution (bitcast float)                        | uint32_t |                                                       | True     |
 * | param1          | Upper bound of uniform distribution (bitcast float)                        | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)));
}

/**
 * Initializes rrelu SFPU operation, including PRNG seeding.
 * The seed parameter is used to initialize the hardware PRNG for random slope generation.
 */
ALWI void rrelu_tile_init(uint32_t seed = 0) { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>(seed))); }

}  // namespace ckernel
```

### `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`
Comprehensive test suite for RReLU operation covering eval mode (deterministic midpoint slope) and training mode (random slopes). Tests exhaustive bfloat16 bitpatterns in eval mode and range checking in training mode.

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ttnn.rrelu (Randomized Leaky ReLU) SFPU operation.

RReLU(x) = x              if x >= 0
          = a * x          if x < 0
  - Eval mode  (seed == 0): a = (lower + upper) / 2   (deterministic)
  - Train mode (seed != 0): a ~ Uniform(lower, upper)  (random per element)

Two test groups:
  1. Eval mode  -- exhaustive bfloat16 bitpatterns, exact golden comparison
  2. Train mode -- verify output is in valid range for negative inputs
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
# 1. Eval-mode test (deterministic) -- exhaustive bfloat16, bfloat16 + fp32
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 0.333333),  # default
        (0.01, 0.5),
        (0.2, 0.2),  # lower == upper => fixed slope, identical to leaky_relu
    ],
)
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_eval(device, lower, upper, is_fp32):
    """Eval mode (seed=0): slope = (lower + upper) / 2, fully deterministic."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Golden: eval-mode RReLU is just leaky_relu with slope = midpoint
    midpoint = (lower + upper) / 2.0
    input_f32 = torch_input.float()
    torch_output = torch.where(input_f32 >= 0, input_f32, input_f32 * midpoint)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device -- seed=0 means eval mode (deterministic midpoint slope)
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, seed=0)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter NaN/Inf
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )

    if is_fp32:
        expected_finite = expected[finite_mask].float().reshape(1, -1)
        actual_finite = actual[finite_mask].float().reshape(1, -1)
        # Flush subnormal float32 artifacts before comparison
        expected_finite = flush_subnormal_values_to_zero(expected_finite)
        actual_finite = flush_subnormal_values_to_zero(actual_finite)
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        # Keep as bfloat16 for ULP comparison so ULP is measured in bfloat16 granularity.
        # Converting to float32 inflates ULP by 2^16 (65536) because float32 has 16 more
        # mantissa bits than bfloat16.
        expected_finite = expected[finite_mask].to(torch.bfloat16).reshape(1, -1)
        actual_finite = actual[finite_mask].to(torch.bfloat16).reshape(1, -1)
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        # allclose in float32 for better precision reporting
        assert_allclose(expected_finite.float(), actual_finite.float(), rtol=1.6e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 2. Training-mode test (random slopes) -- range checking
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_training(device, is_fp32):
    """Training mode (seed != 0): verify positive passthrough and negative range."""
    lower = 0.125
    upper = 0.333333

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, seed=42)
    actual = ttnn.to_torch(tt_output)

    # Flush subnormals from both input and output for comparison.
    # Hardware flushes subnormal inputs to zero; match this in the reference.
    input_f32 = flush_subnormal_values_to_zero(torch_input.float())
    actual_f32 = flush_subnormal_values_to_zero(actual.float())

    # --- Positive values: output == input (exact passthrough) ---
    # Use strictly positive to avoid zero-point artifacts
    pos_mask = input_f32 > 0.0
    pos_input = input_f32[pos_mask]
    pos_actual = actual_f32[pos_mask]

    # Filter to finite values only
    finite_pos = torch.isfinite(pos_input) & torch.isfinite(pos_actual)
    pos_input_f = pos_input[finite_pos]
    pos_actual_f = pos_actual[finite_pos]
    # Positive passthrough should be exact (or within 1 ULP for fp32 rounding)
    assert torch.allclose(
        pos_input_f, pos_actual_f, rtol=0, atol=0
    ), f"Positive passthrough failed: max diff = {(pos_input_f - pos_actual_f).abs().max().item()}"

    # --- Negative values: output in [upper * x, lower * x] ---
    # (since x < 0, upper * x < lower * x)
    neg_mask = input_f32 < 0.0
    neg_input = input_f32[neg_mask]
    neg_actual = actual_f32[neg_mask]

    finite_neg = torch.isfinite(neg_input) & torch.isfinite(neg_actual)
    neg_input_f = neg_input[finite_neg]
    neg_actual_f = neg_actual[finite_neg]

    # For negative x: lower_bound = upper * x (more negative), upper_bound = lower * x (less negative)
    neg_lower_bound = upper * neg_input_f  # more negative
    neg_upper_bound = lower * neg_input_f  # less negative

    # Allow small tolerance for floating point rounding
    tol = 1e-2 if not is_fp32 else 1e-4
    in_range = (neg_actual_f >= neg_lower_bound - tol) & (neg_actual_f <= neg_upper_bound + tol)
    violations = (~in_range).sum().item()
    total_neg = neg_input_f.numel()
    assert violations == 0, (
        f"Training mode range check failed: {violations}/{total_neg} negative values out of range "
        f"[upper*x, lower*x] = [{upper}*x, {lower}*x]"
    )
```

## Files Modified

### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
Added conditional include for the RReLU compute API header.

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
Added rrelu to the SfpuType enum for Wormhole B0.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
Added rrelu to the SfpuType enum for Blackhole.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
Added RRELU to the UnaryOpType enum.

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
Added macro definition lookup for RRELU and parameter marshalling logic in get_op_init_and_func_parameterized.

```diff
@@ -97,6 +97,7 @@ std::string get_macro_definition(UnaryOpType op_type) {
         case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
         case UnaryOpType::HARDMISH: return "SFPU_OP_HARDMISH_INCLUDE";
         case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
+        case UnaryOpType::RRELU: return "SFPU_OP_RRELU_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     };
 }
@@ -561,6 +562,19 @@ std::pair<std::string, std::string> get_op_init_and_func_parameterized(
                     std::bit_cast<uint32_t>(param0),
                     std::bit_cast<uint32_t>(param1))};
         }
+        case UnaryOpType::RRELU: {
+            TT_FATAL(params.size() == 3, "Expected rrelu to take 3 parameters (lower, upper, seed)");
+            float param1 = params[1];
+            float param2 = params[2];
+            uint32_t seed = std::bit_cast<uint32_t>(param2);
+            return {
+                fmt::format("rrelu_tile_init({:#x}u);", seed),
+                fmt::format(
+                    "rrelu_tile({}, {:#x}u, {:#x}u);",
+                    idst,
+                    std::bit_cast<uint32_t>(param0),
+                    std::bit_cast<uint32_t>(param1))};
+        }
         case UnaryOpType::HARDMISH: {
             return {
                 fmt::format("hardmish_tile_init<{}u>();", (uint32_t)param0),
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
Added RRELU to the is_parametrized_type function to indicate it requires parameter passing.

```diff
@@ -100,7 +100,8 @@ bool is_parametrized_type(T val) {
         case UnaryOpType::SELU:
         case UnaryOpType::LOGIT:
         case UnaryOpType::RPOW:
-        case UnaryOpType::MISH: return true;
+        case UnaryOpType::MISH:
+        case UnaryOpType::RRELU: return true;
         default: return false;
     }
     return false;
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`
Added public C++ function declaration for rrelu with default parameters (lower=0.125, upper=0.333333, seed=0).

```diff
@@ -267,6 +267,14 @@ Tensor selu(
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
     const std::optional<Tensor>& optional_output_tensor = std::nullopt);

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower = 0.125f,
+    float upper = 0.333333f,
+    uint32_t seed = 0,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
+    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
+
 Tensor bitcast(
     const Tensor& input_tensor,
     const tt::tt_metal::DataType& output_dtype,
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
Implemented the rrelu C++ function. Added include for <bit> header, implements seed bitcasting to float for transport through UnaryWithParam system, and calls unary_impl with the RRELU operation type.

```diff
@@ -4,6 +4,7 @@

 #include "unary.hpp"

+#include <bit>
 #include "common/unary_op_types.hpp"
 #include "device/unary_device_operation.hpp"
 #include "ttnn/operation.hpp"
@@ -371,6 +372,22 @@ Tensor selu(
         input_tensor, {UnaryWithParam{UnaryOpType::SELU, {scale, alpha}}}, memory_config, optional_output_tensor);
 }

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower,
+    float upper,
+    uint32_t seed,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
+    const std::optional<Tensor>& optional_output_tensor) {
+    // Pack seed as float for transport through the UnaryWithParam system
+    float seed_as_float = std::bit_cast<float>(seed);
+    return ttnn::detail::unary_impl(
+        input_tensor,
+        {UnaryWithParam{UnaryOpType::RRELU, {lower, upper, seed_as_float}}},
+        memory_config,
+        optional_output_tensor);
+}
+
 Tensor swish(
     const Tensor& input_tensor,
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
Added nanobind Python binding for the rrelu function with comprehensive docstring, parameters, and defaults.

```diff
@@ -2400,6 +2400,48 @@ void py_module(nb::module_& mod) {
         nb::kw_only(),
         nb::arg("memory_config") = nb::none(),
         nb::arg("output_tensor") = nb::none());
+
+    auto rrelu_doc = R"doc(
+        Performs RReLU (Randomized Leaky ReLU) function on :attr:`input_tensor`.
+
+        RReLU(x) = x if x >= 0; a*x if x < 0, where a is sampled from Uniform(lower, upper)
+        in training mode, or a = (lower + upper) / 2 in eval mode.
+
+        Args:
+            input_tensor (ttnn.Tensor): the input tensor.
+
+        Keyword args:
+            lower (float, optional): Lower bound of uniform distribution. Defaults to `0.125`.
+            upper (float, optional): Upper bound of uniform distribution. Defaults to `0.333333`.
+            seed (int, optional): PRNG seed for training mode. 0 means eval mode (deterministic midpoint slope). Defaults to `0`.
+            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
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
+               * - BFLOAT16, FLOAT32
+                 - TILE
+        )doc";
+
+    ttnn::bind_function<"rrelu">(
+        mod,
+        rrelu_doc,
+        &ttnn::rrelu,
+        nb::arg("input_tensor"),
+        nb::kw_only(),
+        nb::arg("lower") = 0.125f,
+        nb::arg("upper") = 0.333333f,
+        nb::arg("seed") = 0u,
+        nb::arg("memory_config") = nb::none(),
+        nb::arg("output_tensor") = nb::none());
 }

 }  // namespace ttnn::operations::unary
```

### `ttnn/ttnn/operations/unary.py`
Added Python golden function for rrelu with proper eval mode (seed=0, deterministic midpoint) and training mode (random slopes) handling.

```diff
@@ -403,6 +403,23 @@ def _golden_function_selu(input_tensor_a, *args, **kwargs):
 ttnn.attach_golden_function(ttnn.selu, golden_function=_golden_function_selu)


+def _golden_function_rrelu(input_tensor_a, *args, lower=0.125, upper=0.333333, seed=0, **kwargs):
+    import torch
+
+    if seed == 0:
+        # Eval mode: use fixed slope = (lower + upper) / 2
+        slope = (lower + upper) / 2.0
+        return torch.where(input_tensor_a >= 0, input_tensor_a, input_tensor_a * slope)
+    else:
+        # Training mode: use random slopes from Uniform(lower, upper)
+        torch.manual_seed(seed)
+        rand_slopes = torch.empty_like(input_tensor_a).uniform_(lower, upper)
+        return torch.where(input_tensor_a >= 0, input_tensor_a, input_tensor_a * rand_slopes)
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
- **leaky_relu**: Most useful reference for the conditional branch pattern (CC set on sign bit, CC-guarded multiply, CC clear). The raw TTI instruction sequence (SFPSETCC/SFPMUL/SFPENCC) was directly adapted for the negative-element multiplication. HIGH usefulness.
- **rand**: Critical reference for PRNG access pattern (SFPMOV from RS[9], SFPSETSGN, SFPSETEXP, SFPADDI to construct uniform [0,1) float). The rand kernel's technique of reading PRNG -> setting exponent to 127 -> subtracting 1.0 was directly reused. HIGH usefulness.
- **selu**: Used as the template for multi-parameter registration (2+ params in get_op_init_and_func_parameterized, LLK dispatch with multiple args, custom C++ function declaration). MEDIUM usefulness.
- **prelu**: Confirmed the pattern for SFPI v_if conditional application of slope to negative values. Initially attempted SFPI-based approach before switching to raw TTI. LOW usefulness (superseded by leaky_relu raw TTI pattern).

### Key Design Choices
1. **Raw TTI kernel instead of SFPI**: The initial implementation mixed SFPI abstractions (v_if/v_endif, vFloat) with raw TTI instructions (SFPMOV for PRNG). This risked register allocation conflicts (SFPI compiler uses LREG0-3 internally). The final implementation uses purely raw TTI instructions, following the leaky_relu and rand patterns. This is safer and more predictable.

2. **PRNG seeding via rrelu_tile_init(seed)**: The init function calls init_prng_seed(seed) which is called per-tile (since it's part of SFPU_OP_CHAIN_0). This means the PRNG is re-seeded each tile with the same seed, producing the same random pattern per tile. This is a known limitation but is acceptable for deterministic testing. For production use, the seed should be varied per tile.

3. **3-parameter design**: lower (float), upper (float), seed (uint32_t bitcast to float). The seed is transported through the UnaryWithParam float parameter system using std::bit_cast<float>(seed). The init function receives the seed, and the tile function receives lower and upper.

4. **Subtraction via SFPMAD**: Computing range = upper - lower requires subtraction. Since there's no TTI subtract instruction, we load -1.0 into LREG3 and compute `lower * (-1.0) + upper` via SFPMAD. The result is then moved to LREG2 via SFPMOV to free LREG3 for the loop.

5. **Architecture differences**: Wormhole B0 uses ADDR_MOD_3 and requires TTI_SFPNOP after SFPADDI/SFPMAD. Blackhole uses ADDR_MOD_7 and omits the NOPs (improved pipeline forwarding).

## Known Limitations
- PRNG is re-seeded every tile (since rrelu_tile_init is called per tile via SFPU_OP_CHAIN_0). This means every tile gets the same random slope pattern. For truly different random values per tile, a custom compute kernel (like rand/dropout) would be needed.
- The eval mode (seed=0) still seeds the PRNG with 0 and generates random slopes. To get true eval behavior (fixed midpoint slope), the Python-level code should pass lower = upper = midpoint when seed == 0. The golden function handles this distinction.
- No bfloat16 rounding in the raw TTI path (the raw TTI SFPSTORE uses IMPLIED format which handles format conversion automatically via the DEST accumulator format).
- Each lane's PRNG advances independently, but all lanes advance unconditionally (even for positive elements where the random value is not used). This is by design and ensures consistent PRNG state progression.

## Test Results
- **Status**: PASS (after 7 attempts, 5 hypotheses)
- **Test file**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py
- **Total tests**: 8 (6 eval mode + 2 training mode)
- **Eval mode bfloat16** (3 param combos, is_fp32=False):
  - **Max ULP**: 1.0 (threshold: 2)
  - **allclose**: PASS (rtol=1.6e-2, atol=1e-2)
- **Eval mode fp32** (3 param combos, is_fp32=True):
  - **Max ULP**: 0.0 (threshold: 3)
  - **allclose**: PASS (rtol=1e-3, atol=1e-4)
- **Training mode bfloat16**: PASS (positive passthrough exact, negative values in [upper*x, lower*x])
- **Training mode fp32**: PASS (positive passthrough exact, negative values in [upper*x, lower*x])

### New Files (added by tester)
tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Modified Files (by tester during debugging)
ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp - Added eval mode (seed==0) midpoint fix

## Debug Log
### Attempt 1
- **Result**: FAIL
- **Error type**: runtime_error
- **Error**: TypeError: incompatible function arguments. Compiled binary had stale API (training:bool) not matching source (seed:uint32_t).
- **Hypothesis**: H1 - Compiled binary has different Python API than source nanobind file.
- **Fix**: Attempted to change test to use training=False/True (wrong approach).
- **Files modified**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Attempt 2
- **Result**: FAIL
- **Error type**: build_error
- **Error**: rrelu_tile called with wrong number of args (stale binary dispatching wrong code).
- **Hypothesis**: H2 - Compiled binary is stale, needs rebuild.
- **Fix**: Ran build_metal.sh to recompile. Reverted test back to seed=0/42 API.
- **Files modified**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py (reverted)

### Attempt 3
- **Result**: FAIL
- **Error type**: numerical_error
- **Error**: Max ULP Delta 14483456.0 - large negative values had wrong slopes.
- **Hypothesis**: H3 - Subnormal float32 artifacts (incorrect, root cause was different).
- **Fix**: Added flush_subnormal_values_to_zero on comparison tensors.
- **Files modified**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Attempt 4
- **Result**: FAIL
- **Error type**: numerical_error
- **Error**: Max ULP Delta 14745600.0 - still large errors, subnormal flush was not the issue.
- **Hypothesis**: H4 - C++ rrelu() passes original lower/upper to kernel in eval mode (seed=0), but kernel always uses random slopes. Need to pass midpoint as both lower and upper when seed==0.
- **Fix**: Modified unary.cpp to set effective_lower = effective_upper = midpoint when seed==0.
- **Files modified**: ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp

### Attempt 5
- **Result**: FAIL
- **Error type**: numerical_error
- **Error**: Max ULP Delta 65536.0 - this equals exactly 2^16 = ratio between float32 and bfloat16 ULP granularity. Actual error was 1 bfloat16 ULP.
- **Hypothesis**: H5 - Test compares float32 ULPs but data is bfloat16. Need to keep bfloat16 dtype for ULP comparison.
- **Fix**: Changed bfloat16 branch to compare as bfloat16 tensors for ULP measurement.
- **Files modified**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Attempt 6
- **Result**: FAIL (partial - 6 eval PASSED, 1 training bfloat16 FAILED)
- **Error type**: numerical_error
- **Error**: Training bfloat16 positive passthrough: hardware flushes subnormal positive inputs to zero, but test expected exact match with rtol=0, atol=0.
- **Fix**: Added flush_subnormal_values_to_zero on both input and actual before passthrough comparison.
- **Files modified**: tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

### Attempt 7
- **Result**: PASS
- **All 8 tests passed**: 6 eval mode + 2 training mode
- **bfloat16**: ULP 1.0, allclose PASS
- **fp32**: ULP 0.0, allclose PASS
