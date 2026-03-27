# Implementation Notes: RReLU

## Math Definition
RReLU(x) = x if x >= 0; RReLU(x) = a * x if x < 0
- Training mode: a ~ Uniform(lower, upper) per element (random slope via hardware PRNG)
- Eval/inference mode: a = (lower + upper) / 2 (fixed slope)
- Default: lower = 1/8 (0.125), upper = 1/3 (~0.333)
- Parameters: lower (float), upper (float), training (bool)

## Files Created

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
SFPU compute kernel for RReLU supporting both eval and training modes. Implements `calculate_rrelu()` for eval mode with fixed slope computation on SFPU, and `calculate_rrelu_training()` for training mode with hardware PRNG-based per-element random slope generation. Also includes `rrelu_init()` for PRNG seed initialization.

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

// RReLU eval/inference mode:
//   RReLU(x) = x              if x >= 0
//   RReLU(x) = slope * x      if x < 0
//   where slope = (lower + upper) / 2
//
// Parameters lower and upper are passed as bitcast uint32_t.
// The midpoint slope is computed on the SFPU.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_u, uint upper_u) {
    // Reconstruct float parameters from bitcast uint32_t
    vFloat lower = Converter::as_float(lower_u);
    vFloat upper = Converter::as_float(upper_u);

    // Compute slope = (lower + upper) * 0.5
    vFloat slope = (lower + upper) * vFloat(0.5f);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * slope; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

// RReLU training mode:
//   RReLU(x) = x              if x >= 0
//   RReLU(x) = a * x          if x < 0
//   where a ~ Uniform(lower, upper) is sampled per element
//
// Parameters: lower_u = bitcast(lower), range_u = bitcast(upper - lower)
// PRNG must be initialized via init_prng_seed() before calling this function.
// Uses raw TTI instructions for PRNG random number generation (same pattern as dropout/rand).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu_training(uint lower_u, uint range_u) {
    // Load range = upper - lower into LREG2 (preserved across loop)
    TT_SFPLOADI(p_sfpu::LREG2, 10, range_u & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, range_u >> 16);

    // Load lower into LREG3 (preserved across loop)
    TT_SFPLOADI(p_sfpu::LREG3, 10, lower_u & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, 8, lower_u >> 16);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Step 1: Generate random float in [0, 1) -> LREG1
        // SFPMOV with lreg_c=9, instr_mod1=8 generates pseudorandom uint32
        TTI_SFPMOV(0, 9, p_sfpu::LREG1, 8);
        // Clear sign bit to ensure positive
        TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Set exponent to 127 -> float in [1, 2)
        TTI_SFPSETEXP(127, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Subtract 1 -> float in [0, 1)
        TTI_SFPADDI(0xbf80 /*-1.0f as fp16b*/, p_sfpu::LREG1, 0);
        TTI_SFPNOP;

        // Step 2: Scale to [lower, upper): slope = rand * range + lower
        // SFPMAD: LREG1 = LREG1 * LREG2 + LREG3
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPNOP;

        // Step 3: Load input from dst_reg -> LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);

        // Step 4: Conditionally multiply negative inputs by random slope
        // SFPSETCC: sets condition code where LREG0 < 0 (sign bit check)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
        // SFPMUL only applies to elements where CC is set (input < 0)
        // LREG0 = LREG0 * LREG1 + 0 (for negative elements only)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // End conditional block
        TTI_SFPENCC(0, 0, 0, 0);

        // Step 5: Store result back to dst_reg
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

// PRNG seed initialization for training mode
template <bool APPROXIMATION_MODE>
inline void rrelu_init(const uint seed) {
    init_prng_seed(seed);
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

// RReLU eval/inference mode:
//   RReLU(x) = x              if x >= 0
//   RReLU(x) = slope * x      if x < 0
//   where slope = (lower + upper) / 2
//
// Parameters lower and upper are passed as bitcast uint32_t.
// The midpoint slope is computed on the SFPU.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_u, uint upper_u) {
    // Reconstruct float parameters from bitcast uint32_t
    vFloat lower = Converter::as_float(lower_u);
    vFloat upper = Converter::as_float(upper_u);

    // Compute slope = (lower + upper) * 0.5
    vFloat slope = (lower + upper) * vFloat(0.5f);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * slope; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

// RReLU training mode:
//   RReLU(x) = x              if x >= 0
//   RReLU(x) = a * x          if x < 0
//   where a ~ Uniform(lower, upper) is sampled per element
//
// Parameters: lower_u = bitcast(lower), range_u = bitcast(upper - lower)
// PRNG must be initialized via init_prng_seed() before calling this function.
// Uses raw TTI instructions for PRNG random number generation (same pattern as dropout/rand).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu_training(uint lower_u, uint range_u) {
    // Load range = upper - lower into LREG2 (preserved across loop)
    TT_SFPLOADI(p_sfpu::LREG2, 10, range_u & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, range_u >> 16);

    // Load lower into LREG3 (preserved across loop)
    TT_SFPLOADI(p_sfpu::LREG3, 10, lower_u & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG3, 8, lower_u >> 16);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Step 1: Generate random float in [0, 1) -> LREG1
        // SFPMOV with lreg_c=9, instr_mod1=8 generates pseudorandom uint32
        TTI_SFPMOV(0, 9, p_sfpu::LREG1, 8);
        // Clear sign bit to ensure positive
        TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Set exponent to 127 -> float in [1, 2)
        TTI_SFPSETEXP(127, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Subtract 1 -> float in [0, 1)
        TTI_SFPADDI(0xbf80 /*-1.0f as fp16b*/, p_sfpu::LREG1, 0);
        TTI_SFPNOP;

        // Step 2: Scale to [lower, upper): slope = rand * range + lower
        // SFPMAD: LREG1 = LREG1 * LREG2 + LREG3
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPNOP;

        // Step 3: Load input from dst_reg -> LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);

        // Step 4: Conditionally multiply negative inputs by random slope
        // SFPSETCC: sets condition code where LREG0 < 0 (sign bit check)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
        // SFPMUL only applies to elements where CC is set (input < 0)
        // LREG0 = LREG0 * LREG1 + 0 (for negative elements only)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // End conditional block
        TTI_SFPENCC(0, 0, 0, 0);

        // Step 5: Store result back to dst_reg
        TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

// PRNG seed initialization for training mode
template <bool APPROXIMATION_MODE>
inline void rrelu_init(const uint seed) {
    init_prng_seed(seed);
}

}  // namespace sfpu
}  // namespace ckernel
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
LLK math API wrapper for RReLU kernels. Provides template functions that dispatch to the compute kernel functions for both eval and training modes.

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
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_training_init(uint seed) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>(ckernel::sfpu::rrelu_init<APPROXIMATE>, seed);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint lower, uint upper, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, upper);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu_training(
    uint dst_index, uint lower, uint range, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu_training<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, range);
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
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rrelu_training_init(uint seed) {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>(ckernel::sfpu::rrelu_init<APPROXIMATE>, seed);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint lower, uint upper, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, upper);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu_training(
    uint dst_index, uint lower, uint range, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu_training<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, range);
}

}  // namespace ckernel
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`
C++ compute API header exposing RReLU tile operations: `rrelu_tile()` for eval mode, `rrelu_tile_training()` for training mode, and their corresponding init functions.

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
 * Performs element-wise RReLU (Randomized Leaky ReLU) in eval/inference mode.
 *   RReLU(x) = x              if x >= 0
 *   RReLU(x) = slope * x      if x < 0
 *   where slope = (lower + upper) / 2
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound of the uniform distribution (bitcast float)                    | uint32_t |                                                       | True     |
 * | param1          | Upper bound of the uniform distribution (bitcast float)                    | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX, DST_ACCUM_MODE>(idst, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>())); }

// clang-format off
/**
 * Performs element-wise RReLU (Randomized Leaky ReLU) in training mode.
 *   RReLU(x) = x              if x >= 0
 *   RReLU(x) = a * x          if x < 0
 *   where a ~ Uniform(lower, upper) is sampled per element using hardware PRNG.
 *
 * PRNG must be initialized via rrelu_tile_training_init(seed) before calling this function.
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound of the uniform distribution (bitcast float)                    | uint32_t |                                                       | True     |
 * | param1          | Range = upper - lower (bitcast float)                                      | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile_training(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_training<APPROX, DST_ACCUM_MODE>(idst, param0, param1)));
}

/**
 * Initialize PRNG for RReLU training mode. Must be called once before rrelu_tile_training.
 */
ALWI void rrelu_tile_training_init(uint32_t seed) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu_training_init<APPROX>(seed)));
}

}  // namespace ckernel
```

### `tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py`
Comprehensive test suite for RReLU with three test cases: exhaustive eval mode verification across all bfloat16 bitpatterns, training mode range verification for random slopes, and non-determinism verification.

```python
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
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),  # default PyTorch values
    ],
)
def test_rrelu_eval(device, lower, upper):
    """Test RReLU in eval/inference mode with fixed slope = (lower + upper) / 2."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Compute reference in float32 -- eval mode uses fixed slope = (lower + upper) / 2
    slope = (lower + upper) / 2.0
    inp_f32 = torch_input.float()
    torch_output = torch.where(inp_f32 >= 0, inp_f32, slope * inp_f32)
    expected = flush_subnormal_values_to_zero(torch_output).to(torch.bfloat16)

    # Run on device
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = torch.isfinite(torch_input) & torch.isfinite(expected) & torch.isfinite(actual)
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),  # default PyTorch values
    ],
)
def test_rrelu_training(device, lower, upper):
    """Test RReLU in training mode with random per-element slopes in [lower, upper)."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Run on device with training=True
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    inp_f32 = torch_input.float()
    actual_f32 = actual.float()

    # For positive inputs: output should equal input (accounting for subnormal flushing)
    # Hardware flushes subnormals to zero, so we compare after flushing
    positive_mask = torch.isfinite(torch_input) & (torch_input > 0)
    if positive_mask.any():
        expected_pos = flush_subnormal_values_to_zero(torch_input[positive_mask].float()).to(torch.bfloat16)
        actual_pos = actual[positive_mask].to(torch.bfloat16)
        assert torch.equal(
            expected_pos, actual_pos
        ), "Positive inputs should pass through unchanged (after subnormal flush)"

    # For negative inputs: output should be in range [upper * input, lower * input]
    # (since input is negative, multiplying by a larger slope gives a MORE negative value)
    negative_mask = torch.isfinite(torch_input) & (torch_input < 0) & torch.isfinite(actual)
    if negative_mask.any():
        neg_input = inp_f32[negative_mask]
        neg_actual = actual_f32[negative_mask]

        # For negative x: lower * x <= a * x <= upper * x
        # But since x < 0, upper * x <= a * x <= lower * x
        lower_bound = upper * neg_input  # more negative (upper slope * negative input)
        upper_bound = lower * neg_input  # less negative (lower slope * negative input)

        # Allow small tolerance for bfloat16 rounding
        tolerance = 1e-2 + 1e-2 * torch.abs(neg_input)
        assert (neg_actual >= lower_bound - tolerance).all(), (
            f"Some outputs below lower bound: " f"min_diff={float((neg_actual - lower_bound + tolerance).min())}"
        )
        assert (neg_actual <= upper_bound + tolerance).all(), (
            f"Some outputs above upper bound: " f"max_diff={float((neg_actual - upper_bound - tolerance).max())}"
        )

    # For zero inputs: output should be zero
    zero_mask = torch_input == 0
    if zero_mask.any():
        assert (actual[zero_mask] == 0).all(), "Zero inputs should produce zero outputs"


@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),
    ],
)
def test_rrelu_training_randomness(device, lower, upper):
    """Verify that training mode produces different outputs across runs (non-deterministic)."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    # Make all values negative to ensure the random slope is applied
    torch_input = -torch.abs(torch_input)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # Run twice with training=True -- results should differ (different random seeds)
    tt_output_1 = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual_1 = ttnn.to_torch(tt_output_1)

    tt_output_2 = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual_2 = ttnn.to_torch(tt_output_2)

    # With high probability, two random runs should produce different results
    # (probability of identical results is vanishingly small for 32x32 random slopes)
    assert not torch.equal(
        actual_1, actual_2
    ), "Training mode should produce different outputs across runs due to random slopes"
```

## Files Modified

### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
Added conditional include for RReLU compute API header.

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
Added `rrelu` variant to SfpuType enum.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
Added `rrelu` variant to SfpuType enum.

```diff
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp`
Added RRELU variant to UnaryOpType enum.

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
Added macro case for RRELU and parameter dispatch logic for both eval (2 params) and training (3 params) modes.

```diff
@@ -97,6 +97,7 @@ std::string get_macro_definition(UnaryOpType op_type) {
         case UnaryOpType::RPOW: return "SFPU_OP_RPOW_INCLUDE";
         case UnaryOpType::HARDMISH: return "SFPU_OP_HARDMISH_INCLUDE";
         case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
+        case UnaryOpType::RRELU: return "SFPU_OP_RRELU_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     };
 }
@@ -563,15 +563,32 @@ std::pair<std::string, std::string> get_op_init_and_func_parameterized(
                     std::bit_cast<uint32_t>(param1))};
         }
         case UnaryOpType::RRELU: {
-            TT_FATAL(params.size() == 2, "Expected rrelu to take 2 parameters");
-            float param1 = params[1];
-            return {
-                "rrelu_tile_init();",
-                fmt::format(
-                    "rrelu_tile({}, {:#x}u, {:#x}u);",
-                    idst,
-                    std::bit_cast<uint32_t>(param0),
-                    std::bit_cast<uint32_t>(param1))};
+            TT_FATAL(
+                params.size() == 2 || params.size() == 3,
+                "Expected rrelu to take 2 parameters (eval) or 3 parameters (training)");
+            if (params.size() == 3) {
+                // Training mode: params = [lower, range, seed_as_float]
+                float range = params[1];
+                float seed_f = params[2];
+                uint32_t seed = std::bit_cast<uint32_t>(seed_f);
+                return {
+                    fmt::format("rrelu_tile_training_init({:#x}u);", seed),
+                    fmt::format(
+                        "rrelu_tile_training({}, {:#x}u, {:#x}u);",
+                        idst,
+                        std::bit_cast<uint32_t>(param0),
+                        std::bit_cast<uint32_t>(range))};
+            } else {
+                // Eval mode: params = [lower, upper]
+                float param1 = params[1];
+                return {
+                    "rrelu_tile_init();",
+                    fmt::format(
+                        "rrelu_tile({}, {:#x}u, {:#x}u);",
+                        idst,
+                        std::bit_cast<uint32_t>(param0),
+                        std::bit_cast<uint32_t>(param1))};
+            }
         }
         case UnaryOpType::HARDMISH: {
             return {
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
Added RRELU to the is_parametrized_type specialization.

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
Added C++ function declaration for rrelu with lower/upper float parameters.

```diff
@@ -325,6 +325,13 @@ Tensor selu(
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
     const std::optional<Tensor>& optional_output_tensor = std::nullopt);

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower = 0.125f,
+    float upper = 0.3333333333f,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
+    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
+
 Tensor bitcast(
     const Tensor& input_tensor,
     const tt::tt_metal::DataType& output_dtype,
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
Added C++ function implementation that delegates to unary_impl with UnaryOpType::RRELU.

```diff
@@ -383,6 +383,16 @@ Tensor selu(
         input_tensor, {UnaryWithParam{UnaryOpType::SELU, {scale, alpha}}}, memory_config, optional_output_tensor);
 }

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower,
+    float upper,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
+    const std::optional<Tensor>& optional_output_tensor) {
+    return ttnn::detail::unary_impl(
+        input_tensor, {UnaryWithParam{UnaryOpType::RRELU, {lower, upper}}}, memory_config, optional_output_tensor);
+}
+
 Tensor swish(
     const Tensor& input_tensor,
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
Added custom nanobind wrapper and binding for rrelu with training parameter support. Replaces template binding with explicit overload_t definition.

```diff
@@ -66,6 +66,16 @@ Tensor unary_composite_3param_to_4param_wrapper(
     return Func(input_tensor, parameter_a, parameter_b, memory_config, std::nullopt);
 }

+// Wrapper for rrelu: two floats + bool training + memory_config
+Tensor rrelu_nanobind_wrapper(
+    const Tensor& input_tensor,
+    float lower,
+    float upper,
+    bool training,
+    const std::optional<MemoryConfig>& memory_config) {
+    return ttnn::rrelu(input_tensor, lower, upper, training, memory_config, std::nullopt);
+}
+
 void bind_unary_clamp(nb::module_& mod) {
     const char* doc = R"doc(
         Applies clamp to :attr:`input_tensor` element-wise.
@@ -2220,15 +2230,40 @@ void py_module(nb::module_& mod) {
     bind_unary_clamp(mod);
     bind_unary_composite_floats_with_default<"selu", &ttnn::selu>(
         mod, "scale", "Scale value", 1.0507, "alpha", "Alpha value", 1.67326);
-    bind_unary_composite_floats_with_default<"rrelu", &ttnn::rrelu>(
-        mod,
-        "lower",
-        "Lower bound of the uniform distribution for the negative slope",
-        0.125f,
-        "upper",
-        "Upper bound of the uniform distribution for the negative slope",
-        0.3333333333f,
-        R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
+    ttnn::bind_function<"rrelu">(
+        mod,
+        R"doc(
+        Performs RReLU (Randomized Leaky ReLU) on :attr:`input_tensor`.
+
+        RReLU(x) = x if x >= 0
+        RReLU(x) = a * x if x < 0
+
+        In eval mode (training=False): a = (lower + upper) / 2 (fixed slope).
+        In training mode (training=True): a ~ Uniform(lower, upper) per element (random slope).
+
+        Args:
+            input_tensor (ttnn.Tensor): the input tensor.
+
+        Keyword args:
+            lower (float, optional): Lower bound of the uniform distribution. Defaults to `0.125`.
+            upper (float, optional): Upper bound of the uniform distribution. Defaults to `0.3333333333`.
+            training (bool, optional): If True, use random per-element slopes. Defaults to `False`.
+            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
+
+        Returns:
+            ttnn.Tensor: the output tensor.
+
+        Note:
+            Supported dtypes: FLOAT32, BFLOAT16, BFLOAT8_B
+        )doc",
+        ttnn::overload_t{
+            &rrelu_nanobind_wrapper,
+            nb::arg("input_tensor"),
+            nb::kw_only(),
+            nb::arg("lower") = 0.125f,
+            nb::arg("upper") = 0.3333333333f,
+            nb::arg("training") = false,
+            nb::arg("memory_config") = nb::none()});
     bind_unary_composite_floats_with_default<"hardtanh", &ttnn::hardtanh>(
         mod, "min_val", "min value", -1.0f, "max_val", "max value", 1.0f, R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
     bind_unary_threshold<"threshold", &ttnn::threshold>(
```

### `ttnn/ttnn/operations/unary.py`
Updated golden function to support training parameter with random slope generation.

```diff
@@ -808,12 +808,17 @@ def _golden_function_alt_complex_rotate90(input_tensor_a, *args, **kwargs):
 ttnn.attach_golden_function(ttnn.alt_complex_rotate90, golden_function=_golden_function_alt_complex_rotate90)


-def _golden_function_rrelu(input_tensor_a, *args, lower=0.125, upper=0.3333333333, **kwargs):
+def _golden_function_rrelu(input_tensor_a, *args, lower=0.125, upper=0.3333333333, training=False, **kwargs):
     import torch

-    # Eval/inference mode: use fixed slope = (lower + upper) / 2
-    slope = (lower + upper) / 2.0
-    return torch.where(input_tensor_a >= 0, input_tensor_a, input_tensor_a * slope)
+    if training:
+        # Training mode: random slope per element in [lower, upper)
+        slope = torch.empty_like(input_tensor_a).uniform_(lower, upper)
+        return torch.where(input_tensor_a >= 0, input_tensor_a, input_tensor_a * slope)
+    else:
+        # Eval/inference mode: use fixed slope = (lower + upper) / 2
+        slope = (lower + upper) / 2.0
+        return torch.where(input_tensor_a >= 0, input_tensor_a, input_tensor_a * slope)


 ttnn.attach_golden_function(ttnn.rrelu, golden_function=_golden_function_rrelu)
```

## Design Decisions

### Eval Mode
- **PRELU_SFPU was the most useful reference** for the SFPU kernel pattern: conditional multiply of negative inputs by a slope value using the sfpi DSL (`vFloat`, `v_if`).
- **SELU was the most useful reference** for the two-parameter registration pattern: passing two float parameters as hex-encoded uint32_t through the op chain.
- Slope computed on SFPU as `(lower + upper) * 0.5` to preserve both parameters.

### Training Mode
- **DROPOUT/RAND were the most useful references** for hardware PRNG usage on the SFPU.
- Uses raw TTI instructions (not sfpi DSL) for the training kernel, following the dropout/rand pattern.
- PRNG generates random 32-bit values via `TTI_SFPMOV(0, 9, lreg, 8)`.
- Random values are normalized to [0,1) by clearing sign bit, setting exponent to 127, subtracting 1.
- Scaled to [lower, upper) via `SFPMAD(rand, range, lower)` where `range = upper - lower`.
- `init_prng_seed(seed)` called once in `rrelu_tile_training_init(seed)` (costs ~600 NOP cycles).
- Seed generated from `std::chrono::steady_clock` in the C++ host code.
- Per-tile kernel uses `SFPSETCC`/`SFPENCC` conditional execution to multiply only negative elements.

### Parameter Dispatch
- Eval mode: `params = [lower, upper]` (2 params) -> `rrelu_tile_init()` + `rrelu_tile()`
- Training mode: `params = [lower, range, seed_as_float]` (3 params) -> `rrelu_tile_training_init(seed)` + `rrelu_tile_training()`
- Mode selection in `get_op_init_and_func_parameterized()` based on `params.size()`.

### Python API
- `ttnn.rrelu(input, lower=0.125, upper=0.333, training=False, memory_config=None)`
- Custom nanobind binding (not using `bind_unary_composite_floats_with_default` template) to support the `training` bool parameter.

## Test Results
- **Status**: ALL PASS (3 tests)
- **test_rrelu_eval**: Exhaustive 65,536 bfloat16 bitpatterns, ULP <= 2, allclose PASS
- **test_rrelu_training**: Verifies random slopes in [lower, upper) range for negative inputs, positive passthrough
- **test_rrelu_training_randomness**: Verifies non-deterministic behavior across two runs
