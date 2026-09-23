// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_comp.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ne(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v == s) { v = 0.0f; }
        v_else { v = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v == s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v > s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v < s) { v = 1.0f; }
        v_else { v = 0.0f; }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v < s) { v = 0.0f; }
        v_else { v = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v > s) { v = 0.0f; }
        v_else { v = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = v;

        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// UnaryComp<APPROX, COMP_OP, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode, value)
//   backs unary_ne/eq/gt/ge/lt/le_tile (float compare of each element against the fp32 bit pattern
//   `value`) and unary_*_tile_init. COMP_OP is one of UnaryCompMode::Ne/Eq/Gt/Ge/Lt/Le.
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, UnaryCompMode COMP_OP, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct UnaryComp
    : SfpuUnaryOp<UnaryComp<APPROXIMATION_MODE, COMP_OP, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static_assert(
        COMP_OP == UnaryCompMode::Ne || COMP_OP == UnaryCompMode::Eq || COMP_OP == UnaryCompMode::Gt ||
            COMP_OP == UnaryCompMode::Ge || COMP_OP == UnaryCompMode::Lt || COMP_OP == UnaryCompMode::Le,
        "UnaryComp supports only UnaryCompMode::Ne/Eq/Gt/Ge/Lt/Le");

    static void kernel(uint32_t value) {
        if constexpr (COMP_OP == UnaryCompMode::Ne) {
            calculate_unary_ne<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_OP == UnaryCompMode::Eq) {
            calculate_unary_eq<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_OP == UnaryCompMode::Gt) {
            calculate_unary_gt<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_OP == UnaryCompMode::Ge) {
            calculate_unary_ge<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_OP == UnaryCompMode::Lt) {
            calculate_unary_lt<APPROXIMATION_MODE, ITERATIONS>(value);
        } else {
            calculate_unary_le<APPROXIMATION_MODE, ITERATIONS>(value);
        }
    }
};

}  // namespace sfpu
}  // namespace ckernel
