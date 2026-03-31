# RReLU (Randomized Leaky ReLU) Implementation Notes

## Operation Definition
- **Evaluation mode**: `output = x if x >= 0; output = ((lower + upper) / 2) * x if x < 0`
- **Training mode**: `output = x if x >= 0; output = a * x if x < 0, where a ~ U(lower, upper)`
- **Parameters**: lower (default 0.125), upper (default 1/3), seed (0=eval, non-zero=training)

## Implementation Summary

RReLU is implemented as a standard UnaryOpType going through the UnaryProgramFactory pipeline.
It supports both evaluation mode (deterministic midpoint slope) and training mode (per-element
PRNG-generated random slopes).

### New Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h`
- `tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h`

### Modified Files
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h` (added `rrelu` to SfpuType enum)
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` (added `rrelu` to SfpuType enum)
- `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` (added SFPU_OP_RRELU_INCLUDE guard)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (added RRELU to UnaryOpType enum)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` (added RRELU to is_parametrized_type)
- `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (added get_macro_definition, get_op_init_and_func, string_to_unary_with_param)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` (added rrelu function declaration)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` (added rrelu function implementation)
- `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` (added Python binding via bind_unary_composite_floats_with_default)
- `ttnn/ttnn/operations/unary.py` (added golden function for rrelu)

## Reference Operations Used

1. **leaky_relu** (most useful): Provided the core pattern for conditional multiplication of negative elements using SFPSETCC + CC-guarded SFPMUL + SFPENCC. The eval mode SFPU kernel directly follows this pattern using SFPI abstractions.

2. **prelu** (useful): Demonstrated the SFPI abstraction pattern (`vFloat`, `v_if`, `v_endif`) for the same conditional multiply. The eval mode kernel mirrors prelu's SFPI code.

3. **dropout/rand** (useful for training mode): Provided the PRNG infrastructure pattern:
   - `TTI_SFPMOV(0, 9, LREG, 8)` for hardware PRNG generation
   - `SFPSETSGN` + `SFPSETEXP(127)` + `SFPADDI(-1.0)` for normalizing to [0, 1)
   - `SFPMAD` for scaling to [lower, upper)
   - `init_prng_seed()` for PRNG seeding

4. **selu** (useful): Demonstrated the 2-parameter registration pattern through the unary pipeline (LLK dispatch with params, `get_op_init_and_func` with multiple hex params). Extended to 3 params for rrelu.

## Design Decisions

### Dual-mode SFPU kernel
- **Eval mode**: Uses SFPI abstractions (`vFloat`, `v_if`, `v_endif`) for simplicity and compiler optimization. Computes midpoint = (lower + upper) * 0.5 once, then applies as leaky_relu slope.
- **Training mode**: Uses raw TTI instructions for direct PRNG control. Cannot use SFPI for PRNG because there's no SFPI abstraction for `SFPMOV(PRNG)`.

### PRNG seeding
- The `rrelu_tile_init(seed)` is called per-tile via SFPU_OP_CHAIN_0. A `static bool` guard in the LLK dispatch ensures `init_prng_seed()` (which takes 600 SFPNOP cycles) is only called once.

### Parameter encoding
- 3 float params passed through the standard unary pipeline: lower, upper, seed
- `seed == 0.0f` → evaluation mode (deterministic midpoint)
- `seed != 0.0f` → training mode (seed value bit-cast to uint32 for PRNG)
- The Python API exposes `lower` and `upper` with defaults; the C++ layer currently hardcodes seed=0 (eval mode).

### Address modes
- WH training mode uses `ADDR_MOD_3` (following leaky_relu/dropout convention)
- BH training mode uses `ADDR_MOD_7` (following BH convention)
- Eval mode uses SFPI abstractions which handle address modes automatically

## Known Limitations

1. **Training mode not exposed in Python API**: The current `ttnn.rrelu()` Python binding only supports evaluation mode (seed=0). Training mode with PRNG can be accessed by constructing `UnaryWithParam{UnaryOpType::RRELU, {lower, upper, seed_float}}` directly in C++.

2. **Same seed across all cores**: In training mode, all cores receive the same PRNG seed (baked into compile-time defines). This means all cores start with identical PRNG state. The random patterns will diverge as each core processes different tiles, but initial elements may correlate.

3. **WH pipeline stalls**: The WH training mode kernel includes `TTI_SFPNOP` after `SFPADDI` and `SFPMAD` instructions (required for Wormhole's pipeline). BH does not need these stalls.

## Source Code - New Files

### tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h

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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_u, uint upper_u, uint seed) {
    if (seed == 0) {
        // Evaluation mode: use deterministic midpoint slope = (lower + upper) / 2
        vFloat lower = Converter::as_float(lower_u);
        vFloat upper = Converter::as_float(upper_u);
        vFloat midpoint = (lower + upper) * 0.5f;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat a = dst_reg[0];
            v_if(a < 0.0f) { a = a * midpoint; }
            v_endif;
            dst_reg[0] = a;
            dst_reg++;
        }
    } else {
        // Training mode: per-element random slopes in [lower, upper)
        // Register allocation:
        //   LREG1 = scale = upper - lower
        //   LREG2 = lower
        //   LREG0 = working (input / output)
        //   LREG3 = working (random slope)

        // Load lower into LREG2
        TT_SFPLOADI(p_sfpu::LREG2, 10, lower_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG2, 8, lower_u >> 16);

        // Load upper into LREG1
        TT_SFPLOADI(p_sfpu::LREG1, 10, upper_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, 8, upper_u >> 16);

        // Compute scale = upper - lower: LREG1 = LREG1 * 1.0 + (-lower)
        // First, load -lower into LREG3 (flip sign bit in upper 16 bits)
        TT_SFPLOADI(p_sfpu::LREG3, 10, lower_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG3, 8, (lower_u >> 16) ^ 0x8000);

        // LREG1 = LREG1(upper) * LCONST_1(1.0) + LREG3(-lower) = upper - lower
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG1, 0);
        TTI_SFPNOP;

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            // Load input from DEST
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);

            // Generate random float in [0, 1) using hardware PRNG
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
            // Clear sign bit (force positive)
            TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            // Set exponent to 127 -> value in [1.0, 2.0)
            TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            // Subtract 1.0 to get [0.0, 1.0)
            TTI_SFPADDI(0xbf80 /*-1.0f in bfloat16*/, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // Scale to [lower, upper): slope = rand * scale + lower
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            // Apply slope only to negative elements
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                          // CC <- (LREG0 < 0)
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // x * slope (CC-guarded)
            TTI_SFPENCC(0, 0, 0, 0);                                                       // CC <- ALL_ENABLED

            // Store result back to DEST
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
            dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint32_t seed) {
    if (seed != 0) {
        init_prng_seed(seed);
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rrelu.h

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

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_u, uint upper_u, uint seed) {
    if (seed == 0) {
        // Evaluation mode: use deterministic midpoint slope = (lower + upper) / 2
        vFloat lower = Converter::as_float(lower_u);
        vFloat upper = Converter::as_float(upper_u);
        vFloat midpoint = (lower + upper) * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat a = dst_reg[0];
            v_if(a < 0.0f) { a = a * midpoint; }
            v_endif;
            dst_reg[0] = a;
            dst_reg++;
        }
    } else {
        // Training mode: per-element random slopes in [lower, upper)
        // Register allocation:
        //   LREG1 = scale = upper - lower
        //   LREG2 = lower
        //   LREG0 = working (input / output)
        //   LREG3 = working (random slope)

        // Load lower into LREG2
        TT_SFPLOADI(p_sfpu::LREG2, 10, lower_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG2, 8, lower_u >> 16);

        // Load upper into LREG1
        TT_SFPLOADI(p_sfpu::LREG1, 10, upper_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG1, 8, upper_u >> 16);

        // Compute scale = upper - lower: LREG1 = LREG1 * 1.0 + (-lower)
        // Load -lower into LREG3 (flip sign bit in upper 16 bits)
        TT_SFPLOADI(p_sfpu::LREG3, 10, lower_u & 0xFFFF);
        TT_SFPLOADI(p_sfpu::LREG3, 8, (lower_u >> 16) ^ 0x8000);

        // LREG1 = LREG1(upper) * LCONST_1(1.0) + LREG3(-lower) = upper - lower
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG1, 0);

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            // Load input from DEST
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);

            // Generate random float in [0, 1) using hardware PRNG
            TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);
            // Clear sign bit (force positive)
            TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            // Set exponent to 127 -> value in [1.0, 2.0)
            TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);
            // Subtract 1.0 to get [0.0, 1.0)
            TTI_SFPADDI(0xbf80 /*-1.0f in bfloat16*/, p_sfpu::LREG3, 0);

            // Scale to [lower, upper): slope = rand * scale + lower
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG3, 0);

            // Apply slope only to negative elements
            TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                          // CC <- (LREG0 < 0)
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // x * slope (CC-guarded)
            TTI_SFPENCC(0, 0, 0, 0);                                                       // CC <- ALL_ENABLED

            // Store result back to DEST
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint32_t seed) {
    if (seed != 0) {
        init_prng_seed(seed);
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h

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
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
    // Seed PRNG only once (this init is called per-tile via SFPU_OP_CHAIN)
    static bool prng_seeded = false;
    if (seed != 0 && !prng_seeded) {
        ckernel::sfpu::rrelu_init<APPROXIMATE>(seed);
        prng_seeded = true;
    }
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint lower, uint upper, uint seed, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, upper, seed);
}

}  // namespace ckernel
```

### tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_rrelu.h

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
    llk_math_eltwise_unary_sfpu_init<SfpuType::rrelu, APPROXIMATE>();
    // Seed PRNG only once (this init is called per-tile via SFPU_OP_CHAIN)
    static bool prng_seeded = false;
    if (seed != 0 && !prng_seeded) {
        ckernel::sfpu::rrelu_init<APPROXIMATE>(seed);
        prng_seeded = true;
    }
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rrelu(
    uint dst_index, uint lower, uint upper, uint seed, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rrelu<APPROXIMATE, ITERATIONS>, dst_index, vector_mode, lower, upper, seed);
}

}  // namespace ckernel
```

### tt_metal/hw/inc/api/compute/eltwise_unary/rrelu.h

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
 * Performs element-wise RReLU (Randomized Leaky ReLU) operation.
 * In evaluation mode (seed=0): output = x if x >= 0, output = ((lower + upper) / 2) * x if x < 0
 * In training mode (seed!=0): output = x if x >= 0, output = a * x if x < 0 where a ~ U(lower, upper)
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | Lower bound of the uniform distribution (bit-cast float)                   | uint32_t |                                                       | True     |
 * | param1          | Upper bound of the uniform distribution (bit-cast float)                   | uint32_t |                                                       | True     |
 * | param2          | PRNG seed (0 = eval mode, non-zero = training mode with seed)              | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void rrelu_tile(uint32_t idst, uint32_t param0, uint32_t param1, uint32_t param2) {
    MATH((llk_math_eltwise_unary_sfpu_rrelu<APPROX>(idst, param0, param1, param2)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void rrelu_tile_init(uint32_t seed) { MATH((llk_math_eltwise_unary_sfpu_rrelu_init<APPROX>(seed))); }

}  // namespace ckernel
```

### tests/ttnn/unit_tests/operations/eltwise/test_rrelu.py

```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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


@pytest.mark.parametrize("lower, upper", [(0.125, 1.0 / 3.0)])
def test_rrelu_eval(device, lower, upper):
    """Test evaluation mode with ALL 65536 bfloat16 bit patterns."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Compute reference in float32, flush subnormals to match hardware behavior
    x_f32 = torch_input.float()
    midpoint = (lower + upper) / 2.0
    torch_output = torch.where(x_f32 >= 0, x_f32, midpoint * x_f32)
    expected = flush_subnormal_values_to_zero(torch_output).to(torch.bfloat16)

    # Run on device
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = torch.isfinite(torch_input) & torch.isfinite(expected) & torch.isfinite(actual)
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
```

## Source Code - Modified Files (Diffs)

### tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h

```diff
diff --git a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
index 4f38fb3415..47b413a18b 100644
--- a/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
+++ b/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h

```diff
diff --git a/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h b/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
index 06eee364c8..094758a232 100644
--- a/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
+++ b/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
@@ -154,4 +154,5 @@ enum class SfpuType {
     lerp,
     xielu,
     lgamma,
+    rrelu,
 };
```

### tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h

```diff
diff --git a/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h b/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
index 90f09deb2c..809252cf03 100644
--- a/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
+++ b/tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
@@ -188,6 +188,10 @@
 #include "api/compute/eltwise_unary/lgamma.h"
 #endif

+#if SFPU_OP_RRELU_INCLUDE
+#include "api/compute/eltwise_unary/rrelu.h"
+#endif
+
 #if SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
 #include "api/compute/compute_kernel_api.h"
 #endif
```

### ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
index f189ec8550..aac473508a 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp
@@ -132,6 +132,7 @@ enum class UnaryOpType {
     LOGIT,
     XIELU,
     LGAMMA,
+    RRELU,
 };

 enum class VecMode {
```

### ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
index 48140a6533..1b9ae6b85e 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp
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

### ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
index 5daac04b8a..42b3af70c1 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
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
+            uint32_t seed = std::bit_cast<uint32_t>(static_cast<float>(params[2]));
+            return {
+                fmt::format("rrelu_tile_init({:#x}u);", seed),
+                fmt::format(
+                    "rrelu_tile({}, {:#x}u, {:#x}u, {:#x}u);",
+                    idst,
+                    std::bit_cast<uint32_t>(param0),
+                    std::bit_cast<uint32_t>(param1),
+                    seed)};
+        }
         case UnaryOpType::HARDMISH: {
             return {
                 fmt::format("hardmish_tile_init<{}u>();", (uint32_t)param0),
@@ -884,6 +898,9 @@ UnaryWithParam string_to_unary_with_param(const std::string& name) {
     if (name == "selu") {
         return UnaryWithParam(UnaryOpType::SELU);
     }
+    if (name == "rrelu") {
+        return UnaryWithParam(UnaryOpType::RRELU, {0.125f, 1.0f / 3.0f, 0.0f});
+    }
     if (name == "alt_complex_rotate90") {
         return UnaryWithParam(UnaryOpType::ALT_COMPLEX_ROTATE90);
     }
```

### ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp b/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
index 5f5a57be2c..8c9c594c93 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
@@ -267,6 +267,13 @@ Tensor selu(
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
     const std::optional<Tensor>& optional_output_tensor = std::nullopt);

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower = 0.125f,
+    float upper = 1.0f / 3.0f,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
+    const std::optional<Tensor>& optional_output_tensor = std::nullopt);
+
 Tensor bitcast(
     const Tensor& input_tensor,
     const tt::tt_metal::DataType& output_dtype,
```

### ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp b/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
index cd454c4c2b..0a2ef43208 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp
@@ -371,6 +371,21 @@ Tensor selu(
         input_tensor, {UnaryWithParam{UnaryOpType::SELU, {scale, alpha}}}, memory_config, optional_output_tensor);
 }

+Tensor rrelu(
+    const Tensor& input_tensor,
+    float lower,
+    float upper,
+    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
+    const std::optional<Tensor>& optional_output_tensor) {
+    // Evaluation mode (seed=0): deterministic midpoint slope
+    float seed = 0.0f;
+    return ttnn::detail::unary_impl(
+        input_tensor,
+        {UnaryWithParam{UnaryOpType::RRELU, {lower, upper, seed}}},
+        memory_config,
+        optional_output_tensor);
+}
+
 Tensor swish(
     const Tensor& input_tensor,
     const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
```

### ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp

```diff
diff --git a/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp b/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
index 149f3ab11f..8dc5caadb4 100644
--- a/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
+++ b/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp
@@ -2290,6 +2290,14 @@ void py_module(nb::module_& mod) {
     bind_unary_clamp(mod);
     bind_unary_composite_floats_with_default<"selu", &ttnn::selu>(
         mod, "scale", "Scale value", 1.0507, "alpha", "Alpha value", 1.67326);
+    bind_unary_composite_floats_with_default<"rrelu", &ttnn::rrelu>(
+        mod,
+        "lower",
+        "Lower bound of uniform distribution",
+        0.125f,
+        "upper",
+        "Upper bound of uniform distribution",
+        0.3333333333f);
     bind_unary_composite_floats_with_default<"hardtanh", &ttnn::hardtanh>(
         mod, "min_val", "min value", -1.0f, "max_val", "max value", 1.0f, R"doc(FLOAT32, BFLOAT16, BFLOAT8_B)doc");
     bind_unary_threshold<"threshold", &ttnn::threshold>(
```

### ttnn/ttnn/operations/unary.py

```diff
diff --git a/ttnn/ttnn/operations/unary.py b/ttnn/ttnn/operations/unary.py
index f803730006..099f8a6ea4 100644
--- a/ttnn/ttnn/operations/unary.py
+++ b/ttnn/ttnn/operations/unary.py
@@ -403,6 +403,17 @@ def _golden_function_selu(input_tensor_a, *args, **kwargs):
 ttnn.attach_golden_function(ttnn.selu, golden_function=_golden_function_selu)


+def _golden_function_rrelu(input_tensor_a, *args, lower=0.125, upper=1.0 / 3.0, **kwargs):
+    import torch
+
+    # Evaluation mode: use deterministic midpoint slope = (lower + upper) / 2
+    slope = (lower + upper) / 2.0
+    return torch.where(input_tensor_a >= 0, input_tensor_a, input_tensor_a * slope)
+
+
+ttnn.attach_golden_function(ttnn.rrelu, golden_function=_golden_function_rrelu)
+
+
 def _golden_function_tanhshrink(input_tensor_a, *args, **kwargs):
     import torch
```
