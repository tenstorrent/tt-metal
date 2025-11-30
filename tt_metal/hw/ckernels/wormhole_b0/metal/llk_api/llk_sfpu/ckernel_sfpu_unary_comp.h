// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// template <bool APPROXIMATION_MODE, int ITERATIONS>
// inline void calculate_unary_ne(uint value) {
//     // SFPU microcode
//     sfpi::vFloat s = Converter::as_float(value);

// #pragma GCC unroll 8
//     for (int d = 0; d < ITERATIONS; d++) {
//         sfpi::vFloat v = sfpi::dst_reg[0];
//         v_if(v == s) { v = 0.0f; }
//         v_else { v = 1.0f; }
//         v_endif;

//         sfpi::dst_reg[0] = v;

//         sfpi::dst_reg++;
//     }
// }

// template <bool APPROXIMATION_MODE, int ITERATIONS>
// inline void calculate_unary_eq(uint value) {
//     // SFPU microcode
//     sfpi::vFloat s = Converter::as_float(value);

// #pragma GCC unroll 8
//     for (int d = 0; d < ITERATIONS; d++) {
//         sfpi::vFloat v = sfpi::dst_reg[0];
//         v_if(v == s) { v = 1.0f; }
//         v_else { v = 0.0f; }
//         v_endif;

//         sfpi::dst_reg[0] = v;

//         sfpi::dst_reg++;
//     }
// }

// template <bool APPROXIMATION_MODE, int ITERATIONS>
// inline void calculate_unary_gt(uint value) {
//     // SFPU microcode
//     sfpi::vFloat s = Converter::as_float(value);

// #pragma GCC unroll 8
//     for (int d = 0; d < ITERATIONS; d++) {
//         sfpi::vFloat v = sfpi::dst_reg[0];
//         v_if(v > s) { v = 1.0f; }
//         v_else { v = 0.0f; }
//         v_endif;

//         sfpi::dst_reg[0] = v;

//         sfpi::dst_reg++;
//     }
// }

// template <bool APPROXIMATION_MODE, int ITERATIONS>
// inline void calculate_unary_lt(uint value) {
//     // SFPU microcode
//     sfpi::vFloat s = Converter::as_float(value);

// #pragma GCC unroll 8
//     for (int d = 0; d < ITERATIONS; d++) {
//         sfpi::vFloat v = sfpi::dst_reg[0];
//         v_if(v < s) { v = 1.0f; }
//         v_else { v = 0.0f; }
//         v_endif;

//         sfpi::dst_reg[0] = v;

//         sfpi::dst_reg++;
//     }
// }

// template <bool APPROXIMATION_MODE, int ITERATIONS>
// inline void calculate_unary_ge(uint value) {
//     // SFPU microcode
//     sfpi::vFloat s = Converter::as_float(value);

// #pragma GCC unroll 8
//     for (int d = 0; d < ITERATIONS; d++) {
//         sfpi::vFloat v = sfpi::dst_reg[0];
//         v_if(v < s) { v = 0.0f; }
//         v_else { v = 1.0f; }
//         v_endif;

//         sfpi::dst_reg[0] = v;

//         sfpi::dst_reg++;
//     }
// }

// template <bool APPROXIMATION_MODE, int ITERATIONS>
// inline void calculate_unary_le(uint value) {
//     // SFPU microcode
//     sfpi::vFloat s = Converter::as_float(value);

// #pragma GCC unroll 8
//     for (int d = 0; d < ITERATIONS; d++) {
//         sfpi::vFloat v = sfpi::dst_reg[0];
//         v_if(v > s) { v = 0.0f; }
//         v_else { v = 1.0f; }
//         v_endif;

//         sfpi::dst_reg[0] = v;

//         sfpi::dst_reg++;
//     }
// }

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_unary_comparison(uint value) {
    sfpi::vFloat scalar = Converter::as_float(value);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];

        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(in == scalar) { in = sfpi::vConst0; }
            v_else { in = sfpi::vConst1; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(in == scalar) { in = sfpi::vConst1; }
            v_else { in = sfpi::vConst0; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_gt) {
            v_if(in > scalar) { in = sfpi::vConst1; }
            v_else { in = sfpi::vConst0; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_lt) {
            v_if(in < scalar) { in = sfpi::vConst1; }
            v_else { in = sfpi::vConst0; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_ge) {
            v_if(in < scalar) { in = sfpi::vConst0; }
            v_else { in = sfpi::vConst1; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_le) {
            v_if(in > scalar) { in = sfpi::vConst0; }
            v_else { in = sfpi::vConst1; }
            v_endif;
        }

        sfpi::dst_reg[0] = in;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, int ITERATIONS = 8>
inline void calculate_unary_comparison_uint(uint value) {
    sfpi::vUInt scalar = value;
    sfpi::vUInt one = 1u;
    sfpi::vUInt zero = 0u;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vUInt in = sfpi::dst_reg[0];

        if constexpr (COMP_MODE == SfpuType::unary_ne) {
            v_if(in == scalar) { in = zero; }
            v_else { in = one; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_eq) {
            v_if(in == scalar) { in = one; }
            v_else { in = zero; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_gt) {
            v_if(in > scalar) { in = one; }
            v_else { in = 0u; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_lt) {
            v_if(in < scalar) { in = one; }
            v_else { in = zero; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_ge) {
            v_if(in < scalar) { in = zero; }
            v_else { in = one; }
            v_endif;
        } else if constexpr (COMP_MODE == SfpuType::unary_le) {
            v_if(in > scalar) { in = zero; }
            v_else { in = one; }
            v_endif;
        }

        sfpi::dst_reg[0] = in;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
