# atanh Implementation Notes

## Operation
`atanh(x) = 0.5 * ln((1+x)/(1-x))` for |x| < 1

## Algorithm
The implementation uses IEEE 754 decomposition to compute ln(y) for positive y:
1. Decompose y = 2^e * m, where m in [1, 2) using `exexp` and `setexp` SFPU instructions
2. Approximate ln(m) on [1, 2) using a cubic minimax polynomial: `P(m) = c0 + m*(c1 + m*(c2 + m*c3))`
3. Compute `ln(y) = e * ln(2) + P(m)`

atanh is then: `0.5 * (ln(1+x) - ln(1-x))`

Polynomial coefficients (from rpow scalar log2 precomputation):
- c0 = -0x1.952992p+0f (~-1.5828)
- c1 = 0x2.4f5388p+0f (~2.3110)
- c2 = -0xd.e712ap-4f (~-0.8691)
- c3 = 0x2.44734p-4f (~0.1416)

c0, c1, c2 are stored in programmable constant registers (set during init).
c3 is loaded as an immediate.

## Which Reference Operations Were Most Useful and Why
1. **hardsigmoid** - Most useful for the overall file structure (API header, LLK dispatch, ckernel_sfpu, sfpu_split_includes pattern). It's a clean, simple non-parameterized unary op that served as the template for all abstraction layers.
2. **cbrt** - Showed how to use programmable constant registers (`vConstFloatPrgm0/1/2`) for polynomial coefficients and how to pass an init function to `llk_math_eltwise_unary_sfpu_init`.
3. **rpow** - Provided the cubic polynomial coefficients for ln(m) on [1, 2) and showed usage of `exexp`, `setexp`, and `int32_to_float` SFPU instructions for IEEE 754 decomposition.
4. **softshrink** and **hardtanh** - Confirmed parameterized vs non-parameterized patterns in `unary_op_utils.cpp`.

## Deviations from Standard Patterns
- The SFPU kernel is more complex than typical unary ops (hardsigmoid, hardtanh) because atanh requires computing two natural logarithms from scratch using SFPI instructions. Standard log/reciprocal primitives were intentionally removed, so IEEE 754 bit decomposition + polynomial approximation was used.
- Uses all 3 programmable constant registers for polynomial coefficients, following the cbrt pattern.

## Known Limitations or Concerns
- **Accuracy**: The cubic polynomial for ln(m) provides ~2-3 decimal digits of accuracy, sufficient for bfloat16 (~2.1 decimal digits). For fp32 accumulation mode, accuracy may be insufficient for the full mantissa precision.
- **Edge cases**: Values very close to x = +/-1 produce large outputs (atanh approaches +/-infinity). The polynomial approximation for ln near zero may lose precision for inputs like 1-x when x is very close to 1.
- **No fp32/fp16b path split**: Unlike cbrt, this kernel does not branch on `is_fp32_dest_acc_en`. A single code path is used for both accumulation modes. This is acceptable for bfloat16 precision targets.
- **Register pressure**: Each loop iteration uses many intermediates (two full ln computations), but the SFPI compiler handles register allocation automatically with `#pragma GCC unroll 8`.

---

## Source Code

### New Files

#### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`

> Blackhole variant is identical (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_atanh.h`).

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// atanh(x) = 0.5 * ln((1+x)/(1-x)) = 0.5 * (ln(1+x) - ln(1-x))
// Valid for |x| < 1.
//
// ln(y) is computed via IEEE 754 decomposition:
//   y = 2^e * m, where m in [1, 2)
//   ln(y) = e * ln(2) + P(m)
// where P(m) is a cubic minimax polynomial approximation for ln(m) on [1, 2).
// Coefficients are from the rpow scalar log2 precomputation (Horner form):
//   P(m) = c0 + m * (c1 + m * (c2 + m * c3))
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_atanh() {
    constexpr float c3 = 0x2.44734p-4f;  // ~0.1416
    constexpr float ln2 = 0.6931471805599453f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // a = 1 + x, b = 1 - x (both positive for |x| < 1)
        sfpi::vFloat a = x + sfpi::vConst1;
        sfpi::vFloat b = -x + sfpi::vConst1;

        // ln(a): decompose a = 2^ea * ma, ma in [1, 2)
        sfpi::vInt ea = sfpi::exexp(a);
        sfpi::vFloat ma = sfpi::setexp(a, 127);
        // P(ma) = c0 + ma*(c1 + ma*(c2 + ma*c3))
        sfpi::vFloat pa = ma * c3 + sfpi::vConstFloatPrgm2;
        pa = pa * ma + sfpi::vConstFloatPrgm1;
        pa = pa * ma + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_a = sfpi::int32_to_float(ea, 0) * ln2 + pa;

        // ln(b): decompose b = 2^eb * mb, mb in [1, 2)
        sfpi::vInt eb = sfpi::exexp(b);
        sfpi::vFloat mb = sfpi::setexp(b, 127);
        sfpi::vFloat pb = mb * c3 + sfpi::vConstFloatPrgm2;
        pb = pb * mb + sfpi::vConstFloatPrgm1;
        pb = pb * mb + sfpi::vConstFloatPrgm0;
        sfpi::vFloat ln_b = sfpi::int32_to_float(eb, 0) * ln2 + pb;

        // atanh(x) = 0.5 * (ln(a) - ln(b))
        sfpi::vFloat result = (ln_a - ln_b) * 0.5f;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void atanh_init() {
    // Cubic polynomial coefficients for ln(m) on [1, 2)
    sfpi::vConstFloatPrgm0 = -0x1.952992p+0f;  // c0 ~ -1.5828
    sfpi::vConstFloatPrgm1 = 0x2.4f5388p+0f;   // c1 ~  2.3110
    sfpi::vConstFloatPrgm2 = -0xd.e712ap-4f;   // c2 ~ -0.8691
}

}  // namespace ckernel::sfpu
```

#### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`

> Blackhole variant is identical (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_atanh.h`).

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
    llk_math_eltwise_unary_sfpu_init<SfpuType::atanh, APPROXIMATE>(sfpu::atanh_init<APPROXIMATE>);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_atanh(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_atanh<APPROXIMATE, ITERATIONS>, dst_index, vector_mode);
}

}  // namespace ckernel
```

#### `tt_metal/hw/inc/api/compute/eltwise_unary/atanh.h`

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

// clang-format off
 /**
 * Performs element-wise inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x)).
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
ALWI void atanh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_atanh<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void atanh_tile_init() { MATH((llk_math_eltwise_unary_sfpu_atanh_init<APPROX>())); }

}  // namespace ckernel
```

### Modified Files

#### `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`

> Same change in Blackhole variant.

```diff
 enum class SfpuType {
     unused = 0,
     hardsigmoid,
     hardtanh,
     hardswish,
     softshrink,
+    atanh,
 };
```

#### `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

```diff
 #if SFPU_OP_SOFTSHRINK_INCLUDE
 #include "api/compute/eltwise_unary/softshrink.h"
 #endif
+
+#if SFPU_OP_ATANH_INCLUDE
+#include "api/compute/eltwise_unary/atanh.h"
+#endif
```

#### `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

```diff
 std::string get_macro_definition(UnaryOpType op_type) {
     switch (op_type) {
         case UnaryOpType::HARDTANH: return "SFPU_OP_HARDTANH_INCLUDE";
         case UnaryOpType::HARDSWISH: return "SFPU_OP_HARDSWISH_INCLUDE";
         case UnaryOpType::SOFTSHRINK: return "SFPU_OP_SOFTSHRINK_INCLUDE";
+        case UnaryOpType::ATANH: return "SFPU_OP_ATANH_INCLUDE";
         default: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
     };
 }
```

```diff
 std::pair<std::string, std::string> get_op_init_and_func_default(
     UnaryOpType op_type, std::string idst, ...) {
     switch (op_type) {
         case UnaryOpType::HARDSIGMOID: return {"hardsigmoid_tile_init();", ...};
         case UnaryOpType::HARDSWISH: return {"hardswish_tile_init();", ...};
+        case UnaryOpType::ATANH: return {"atanh_tile_init();", fmt::format("atanh_tile({});", idst)};
         default: TT_THROW("unexpected op type {}", op_type);
     };
 }
```
