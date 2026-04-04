# Implementation Notes: lgamma

## Math Definition
- **Operation**: lgamma (log gamma)
- **Formula**: ln(|Gamma(x)|)
- **Method**: Lanczos approximation with g=5 (Numerical Recipes coefficients)
- **Detailed Formula**:
  ```
  lgamma(x) = 0.5*ln(2*pi) + (x - 0.5)*ln(x + 4.5) - (x + 4.5) + ln(series)
  ```
  where:
  ```
  series = 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3)
  ```
  Using the first 4 Lanczos coefficients:
  - c1 = 76.18009172947146
  - c2 = -86.50532032941677
  - c3 = 24.01409824083091
  - c4 = -1.231739572450155

## Algorithm
The implementation uses the Lanczos approximation matching the existing composite implementation in `unary_composite_op.cpp`. The SFPU kernel computes:

1. **Lanczos Series Calculation**: Computes 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3) using four reciprocal calls
2. **Logarithm Calls**: Computes ln(x + 4.5) and ln(series) using the log helper
3. **Final Combination**: Assembles the result from intermediate terms
4. **Special Case Handling**: lgamma(1) = 0 and lgamma(2) = 0 are handled explicitly via `v_if` conditionals

### SFPU Helpers Used
- `_sfpu_reciprocal_<1>` from `sfpu/ckernel_sfpu_recip.h` - 1 Newton-Raphson iteration (bfloat16 sufficient)
- `_calculate_log_body_no_init_` from `sfpu/ckernel_sfpu_log.h` - inline-constant variant (no programmable register conflict)

### Init Function
`lgamma_init()` calls `_init_sfpu_reciprocal_<APPROXIMATION_MODE>()` to set up the 3 programmable constant registers (vConstFloatPrgm0/1/2) needed by the reciprocal function.

### Special Cases
- lgamma(1) = 0 and lgamma(2) = 0 are handled explicitly via `v_if` conditionals.
- Negative x values are not handled by the Lanczos formula (the approximation is only valid for x > 0). The reflection formula would be needed for full support but is too complex for SFPU.

## Design Decisions

### Reference Operations Used
- **hardsigmoid**: Most useful for the overall file structure pattern - simple no-parameter unary op with identical WH/BH implementations. Used as the template for API header, LLK dispatch, and registration patterns.
- **selu**: Useful for understanding how `_init_sfpu_reciprocal_` and `_calculate_exponential_piecewise_` are called from SFPU kernels. Showed the pattern for including `sfpu/ckernel_sfpu_exp.h` (which transitively includes recip).
- **cosh**: Showed how to include tt_llk helper functions (`sfpu/ckernel_sfpu_exp.h`).
- **cbrt**: Showed the programmable constant register pattern and `#pragma GCC unroll` usage.

### Deviations from Standard Patterns
- Uses `#pragma GCC unroll 0` (no unrolling) instead of `#pragma GCC unroll 8` because the lgamma kernel body is very large (4 reciprocal calls + 2 log calls + series accumulation + conditionals), and unrolling 8x would cause excessive code size.
- Includes two tt_llk helper headers (`sfpu/ckernel_sfpu_log.h` and `sfpu/ckernel_sfpu_recip.h`) whereas most simple ops only include `ckernel.h` and `ckernel_defs.h`. This is necessary because lgamma is a composite function requiring log and reciprocal building blocks.

## Files Created

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h`
Core SFPU kernel implementation for Wormhole B0 with the `calculate_lgamma()` template function computing the Lanczos approximation and `lgamma_init()` initialization function. Identical copy at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h`.

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

// lgamma(x) = ln(|Gamma(x)|)
// Uses Lanczos approximation with g=5 (Numerical Recipes coefficients).
// lgamma(x) = 0.5*ln(2*pi) + (x - 0.5)*ln(x + 4.5) - (x + 4.5) + ln(series)
// where series = 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3)
// Valid for x > 0. Special cases: lgamma(1) = lgamma(2) = 0.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lgamma() {
    constexpr float half_ln_2pi = 0.918938531357171f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Lanczos series: 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3)
        sfpi::vFloat series = sfpi::vConst1;
        series = series + 76.18009172947146f * _sfpu_reciprocal_<1>(x);
        series = series + -86.50532032941677f * _sfpu_reciprocal_<1>(x + sfpi::vConst1);
        series = series + 24.01409824083091f * _sfpu_reciprocal_<1>(x + 2.0f);
        series = series + -1.231739572450155f * _sfpu_reciprocal_<1>(x + 3.0f);

        sfpi::vFloat t = x + 4.5f;
        sfpi::vFloat log_t = _calculate_log_body_no_init_(t);
        sfpi::vFloat log_series = _calculate_log_body_no_init_(series);

        // result = (x - 0.5) * log(t) - t + 0.5*ln(2*pi) + log(series)
        sfpi::vFloat result = (x - 0.5f) * log_t - t + half_ln_2pi + log_series;

        // Special cases: lgamma(1) = 0, lgamma(2) = 0
        v_if(x == sfpi::vConst1) { result = 0.0f; }
        v_endif;
        v_if(x == 2.0f) { result = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void lgamma_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h`
Core SFPU kernel implementation for Blackhole. Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lgamma.h`.

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

// lgamma(x) = ln(|Gamma(x)|)
// Uses Lanczos approximation with g=5 (Numerical Recipes coefficients).
// lgamma(x) = 0.5*ln(2*pi) + (x - 0.5)*ln(x + 4.5) - (x + 4.5) + ln(series)
// where series = 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3)
// Valid for x > 0. Special cases: lgamma(1) = lgamma(2) = 0.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lgamma() {
    constexpr float half_ln_2pi = 0.918938531357171f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Lanczos series: 1 + c1/x + c2/(x+1) + c3/(x+2) + c4/(x+3)
        sfpi::vFloat series = sfpi::vConst1;
        series = series + 76.18009172947146f * _sfpu_reciprocal_<1>(x);
        series = series + -86.50532032941677f * _sfpu_reciprocal_<1>(x + sfpi::vConst1);
        series = series + 24.01409824083091f * _sfpu_reciprocal_<1>(x + 2.0f);
        series = series + -1.231739572450155f * _sfpu_reciprocal_<1>(x + 3.0f);

        sfpi::vFloat t = x + 4.5f;
        sfpi::vFloat log_t = _calculate_log_body_no_init_(t);
        sfpi::vFloat log_series = _calculate_log_body_no_init_(series);

        // result = (x - 0.5) * log(t) - t + 0.5*ln(2*pi) + log(series)
        sfpi::vFloat result = (x - 0.5f) * log_t - t + half_ln_2pi + log_series;

        // Special cases: lgamma(1) = 0, lgamma(2) = 0
        v_if(x == sfpi::vConst1) { result = 0.0f; }
        v_endif;
        v_if(x == 2.0f) { result = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void lgamma_init() {
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
```

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_lgamma.h`
LLK dispatch wrapper for Wormhole B0 that provides the template-based interface `llk_math_eltwise_unary_sfpu_lgamma_init()` and `llk_math_eltwise_unary_sfpu_lgamma()` for calling the core kernel. Identical copy at `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_lgamma.h`.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_lgamma.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lgamma_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::lgamma, APPROXIMATE>(ckernel::sfpu::lgamma_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_lgamma(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_lgamma<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_lgamma.h`
LLK dispatch wrapper for Blackhole. Identical copy at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_lgamma.h`.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_lgamma.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lgamma_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::lgamma, APPROXIMATE>(ckernel::sfpu::lgamma_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_lgamma(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_lgamma<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h`
Public C++ API header providing user-facing `lgamma_tile()` and `lgamma_tile_init()` functions. Wraps the conditional compilation logic around LLK dispatch calls.

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_lgamma.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise lgamma operation: ln(|Gamma(x)|).
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
ALWI void lgamma_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_lgamma<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void lgamma_tile_init() { MATH((llk_math_eltwise_unary_sfpu_lgamma_init<APPROX>())); }

}  // namespace ckernel
```

## Files Modified

### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`
Added `lgamma` to the `SfpuType` enum to register the operation as a known SFPU type.

```diff
@@ -11,4 +11,5 @@
     hardsigmoid,
     selu,
     hardtanh,
+    lgamma,
 };
```

### `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h`
Added `lgamma` to the `SfpuType` enum to register the operation as a known SFPU type.

```diff
@@ -11,4 +11,5 @@
     hardsigmoid,
     selu,
     hardtanh,
+    lgamma,
 };
```

### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`
Added conditional include of the public API header based on the `SFPU_OP_LGAMMA_INCLUDE` preprocessor flag.

```diff
@@ -23,3 +23,7 @@
 #if SFPU_OP_HARDTANH_INCLUDE
 #include "api/compute/eltwise_unary/hardtanh.h"
 #endif
+
+#if SFPU_OP_LGAMMA_INCLUDE
+#include "api/compute/eltwise_unary/lgamma.h"
+#endif
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
Added lgamma case to the `get_compute_kernel_define()` function to map `UnaryOpType::LGAMMA` to the preprocessor flag, and added the init/kernel call pattern in the `get_init_and_compute_calls()` function.

```diff
@@ -21,6 +21,7 @@
         case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";
         case UnaryOpType::SELU: return "SFPU_OP_SELU_INCLUDE";
         case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
+        case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     };
 }
@@ -65,6 +66,7 @@
         case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
         case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
         case UnaryOpType::SELU: return {"selu_tile_init();", fmt::format("selu_tile({});", idst)};
+        case UnaryOpType::LGAMMA: return {"lgamma_tile_init();", fmt::format("lgamma_tile({});", idst)};
         default: TT_THROW("unexpected op type {}", op_type);
     };
 }
```

### `ttnn/cpp/ttnn/operations/eltwise/unary_ng/common/unary_ng_op_utils.cpp`
Reorganized the `get_compute_kernel_define()` function to add lgamma case and refactored the `get_init_and_compute_calls()` function to move lgamma from a stub case into a proper implementation case.

```diff
@@ -20,6 +20,7 @@
     switch (op_type) {
         case UnaryOpType::COSH: return "SFPU_OP_COSH_INCLUDE";
         case UnaryOpType::CBRT: return "SFPU_OP_CBRT_INCLUDE";
+        case UnaryOpType::LGAMMA: return "SFPU_OP_LGAMMA_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     }
 }
@@ -87,8 +88,8 @@
         case UnaryOpType::TRUNC: return {"rounding_op_tile_init();", fmt::format("trunc_tile({});", idst)};
         case UnaryOpType::FRAC: return {"rounding_op_tile_init();", fmt::format("frac_tile({});", idst)};
         case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
-        case UnaryOpType::HARDSWISH:
-        case UnaryOpType::LGAMMA: return {};
+        case UnaryOpType::HARDSWISH: return {};
+        case UnaryOpType::LGAMMA: return {"lgamma_tile_init();", fmt::format("lgamma_tile({});", idst)};
         case UnaryOpType::SOFTSIGN: return {"softsign_tile_init();", fmt::format("softsign_tile({});", idst)};
         case UnaryOpType::CBRT: return {"cbrt_tile_init();", fmt::format("cbrt_tile({});", idst)};
         case UnaryOpType::COSH: return {"cosh_tile_init();", fmt::format("cosh_tile({});", idst)};
```

### `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp`
Added Python binding for the lgamma operation using `bind_unary_operation<"lgamma">()`.

```diff
@@ -1799,6 +1799,12 @@
         R"doc(\text{selu}(x) = \text{scale} \times (\max(0, x) + \min(0, \alpha \times (\exp(x) - 1))))doc",
         "",
         R"doc(BFLOAT16, BFLOAT8_B, FLOAT32)doc");
+
+    bind_unary_operation<"lgamma", &ttnn::lgamma>(
+        mod,
+        R"doc(\mathrm{{output\_tensor}}_i = \ln(|\Gamma(\mathrm{{input\_tensor}}_i)|))doc",
+        "",
+        R"doc(BFLOAT16, FLOAT32)doc");
     {
         auto doc = fmt::format(
             R"doc(
```

### `ttnn/ttnn/operations/unary.py`
Added lgamma to the reference function mapping and to the TTNN C++ function registration list.

```diff
@@ -41,6 +41,7 @@
             "cbrt": torch_cbrt,
             "hardsigmoid": torch.nn.functional.hardsigmoid,
             "selu": lambda _x: torch.nn.functional.selu(_x.to(torch.float)),
+            "lgamma": torch.lgamma,
         }

         golden_keys = set(name_to_golden_function.keys())
@@ -61,6 +62,7 @@
     ttnn.cbrt,
     ttnn.hardsigmoid,
     ttnn.selu,
+    ttnn.lgamma,
 ]
 for unary_function in TTNN_ELTWISE_UNARY_CPP_FUNCTIONS:
     register_ttnn_cpp_unary_function(unary_function)
```

## Known Limitations

1. **Positive x only**: The Lanczos approximation is valid for x > 0. Negative non-integer inputs would need the reflection formula (lgamma(x) = ln(pi/|sin(pi*x)|) - lgamma(1-x)), which requires a sin(pi*x) implementation not available as an SFPU helper.

2. **Precision**: Using `_sfpu_reciprocal_<1>` (1 Newton iteration) gives bfloat16-level precision. For float32 accuracy, `_sfpu_reciprocal_<2>` would be needed, but register pressure may be an issue.

3. **Performance**: The kernel is computation-heavy (~4 reciprocals + 2 logs per element). This is inherent to the lgamma function's complexity.
