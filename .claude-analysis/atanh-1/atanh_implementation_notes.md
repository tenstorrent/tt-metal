# atanh Implementation Notes

## Math Definition
atanh(x) = 0.5 * ln((1+x)/(1-x)) for |x| < 1 (inverse hyperbolic tangent)

## Implementation Strategy
The implementation follows the softsign pattern (LLK dispatch layer with separate SFPU kernel file).

The SFPU kernel computes atanh using:
1. Compute numerator: 1 + x
2. Compute denominator: 1 - x
3. Compute reciprocal of denominator using `_sfpu_reciprocal_<2>()`
4. Multiply to get ratio: (1+x) / (1-x)
5. Compute natural log using `_calculate_log_body_no_init_()`
6. Multiply by 0.5

## Reference Operations Used
- **softsign**: LLK dispatch pattern, reciprocal usage, compute API structure
- **acosh/asinh** (trigonometry.h): Same inverse hyperbolic family, `_calculate_log_body_no_init_()` usage
- **log**: Core log implementation via Chebyshev approximation
- **cosh**: Compute API macros pattern
- **selu**: Conditional SFPU logic pattern

## Key Design Decisions
1. Used `_sfpu_reciprocal_<2>()` for division (2 Newton-Raphson iterations for fp32 precision)
2. Used `_calculate_log_body_no_init_()` from `ckernel_sfpu_log.h` for the natural log (same as acosh/asinh)
3. Init function calls `_init_sfpu_reciprocal_<>()` to set up reciprocal constants
4. No special boundary handling needed - the hardware handles edge cases naturally:
   - x = 0: atanh(0) = 0.5 * ln(1/1) = 0.5 * 0 = 0
   - x -> 1: ratio -> inf, ln(inf) -> inf
   - x -> -1: ratio -> 0, ln(0) -> -inf

## Deviations from Standard Patterns
None - follows the exact same pattern as softsign and the inverse hyperbolic functions in trigonometry.h.

### New Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h
tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h
tests/ttnn/unit_tests/operations/eltwise/test_atanh.py

### Modified Files
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h
tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp
ttnn/ttnn/experimental_loader/golden_functions.py

## Known Limitations
- Input must be in range |x| < 1 for mathematically valid results
- Precision may be limited by the 3rd-order Chebyshev approximation used in the log implementation
- For values very close to +/-1, numerical precision degrades as the ratio approaches infinity/zero

## Test Results
All 4 tests passed on first attempt (no iterations needed):
- test_atanh_bfloat16[w=64-h=32]: PASSED
- test_atanh_fp32[w=64-h=32]: PASSED
- test_atanh_zero[w=64-h=32]: PASSED
- test_atanh_small_values[w=64-h=32]: PASSED

## Source Code

### SFPU Kernel: ckernel_sfpu_atanh.h (wormhole_b0 and blackhole - identical)

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1 + x) / (1 - x))  for |x| < 1
// Implementation: compute (1+x), compute (1-x), compute reciprocal of (1-x),
// multiply to get ratio, compute ln, multiply by 0.5
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Compute numerator: 1 + x
        sfpi::vFloat num = x + sfpi::vConst1;

        // Compute denominator: 1 - x
        sfpi::vFloat den = sfpi::vConst1 - x;

        // Compute reciprocal of denominator: 1 / (1 - x)
        sfpi::vFloat recip_den = _sfpu_reciprocal_<2>(den);

        // Compute ratio: (1 + x) / (1 - x)
        sfpi::vFloat ratio = num * recip_den;

        // Compute ln((1 + x) / (1 - x))
        sfpi::vFloat log_val = _calculate_log_body_no_init_(ratio);

        // Multiply by 0.5
        sfpi::dst_reg[0] = log_val * 0.5f;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
```

### LLK Dispatch: llk_math_eltwise_unary_sfpu_atanh.h (wormhole_b0 and blackhole - identical)

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_atanh.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_atanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(ckernel::sfpu::atanh_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### Compute API: atanh.h

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_atanh.h"
#endif

namespace ckernel {

ALWI void atanh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst))); }

ALWI void atanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_atanh_init<APPROX>())); }

}  // namespace ckernel
```

### Diffs for Modified Files

#### llk_sfpu_types.h (both wormhole_b0 and blackhole)
```diff
     hardswish,
     softshrink,
+    atanh,
 };
```

#### sfpu_split_includes.h
```diff
 #if SFPU_OP_SOFTSHRINK_INCLUDE
 #include "api/compute/eltwise_unary/softshrink.h"
 #endif
+
+#if SFPU_OP_ATANH_INCLUDE
+#include "api/compute/eltwise_unary/atanh.h"
+#endif
```

#### unary_op_utils.cpp - get_macro_definition
```diff
         case UnaryOpType::SOFTSHRINK: return "SFPU_OP_SOFTSHRINK_INCLUDE";
+        case UnaryOpType::ATANH: return "SFPU_OP_ATANH_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
```

#### unary_op_utils.cpp - get_op_init_and_func_default
```diff
         case UnaryOpType::LGAMMA: return {"lgamma_tile_init();", fmt::format("lgamma_tile({});", idst)};
+        case UnaryOpType::ATANH: return {"atanh_tile_init();", fmt::format("atanh_tile({});", idst)};
         default: TT_THROW("unexpected op type {}", op_type);
```

#### golden_functions.py
```diff
+def _atanh_golden_function(input_tensor, *args, **kwargs):
+    import torch
+    return torch.atanh(input_tensor)
+
+if hasattr(ttnn, "atanh"):
+    ttnn.attach_golden_function(ttnn.atanh, _atanh_golden_function)
```
