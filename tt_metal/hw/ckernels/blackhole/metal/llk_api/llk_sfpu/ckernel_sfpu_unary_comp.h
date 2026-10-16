// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_comp.h"

namespace ckernel {
namespace sfpu {

inline void unary_ne_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ne(std::uint32_t value) {
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

inline void unary_eq_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(std::uint32_t value) {
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

inline void unary_gt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(std::uint32_t value) {
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

inline void unary_lt_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(std::uint32_t value) {
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

inline void unary_ge_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(std::uint32_t value) {
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

inline void unary_le_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(std::uint32_t value) {
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

// Op class for comparing a float tile in Dest against a scalar: x OP value ? 1.0 : 0.0, with value
// passed as the bits of a float.
template <bool APPROXIMATION_MODE, CompareOp COMP_MODE, int ITERATIONS = 8>
struct UnaryComp : SfpuUnaryOp<UnaryComp<APPROXIMATION_MODE, COMP_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(std::uint32_t value) {
        if constexpr (COMP_MODE == CompareOp::eq) {
            calculate_unary_eq<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_MODE == CompareOp::ne) {
            calculate_unary_ne<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_MODE == CompareOp::lt) {
            calculate_unary_lt<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_MODE == CompareOp::le) {
            calculate_unary_le<APPROXIMATION_MODE, ITERATIONS>(value);
        } else if constexpr (COMP_MODE == CompareOp::gt) {
            calculate_unary_gt<APPROXIMATION_MODE, ITERATIONS>(value);
        } else {
            calculate_unary_ge<APPROXIMATION_MODE, ITERATIONS>(value);
        }
    }
    static inline __attribute__((always_inline)) void init_op() {
        if constexpr (COMP_MODE == CompareOp::eq) {
            unary_eq_init();
        } else if constexpr (COMP_MODE == CompareOp::ne) {
            unary_ne_init();
        } else if constexpr (COMP_MODE == CompareOp::lt) {
            unary_lt_init();
        } else if constexpr (COMP_MODE == CompareOp::le) {
            unary_le_init();
        } else if constexpr (COMP_MODE == CompareOp::gt) {
            unary_gt_init();
        } else {
            unary_ge_init();
        }
    }
};

}  // namespace sfpu
}  // namespace ckernel
