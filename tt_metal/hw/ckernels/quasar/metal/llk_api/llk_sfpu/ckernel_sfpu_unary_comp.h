// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

inline void unary_ne_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ne(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Store the default, then overwrite the taken lanes under the predicate: two
        // SFPSTOREs of the constant registers instead of two SFPMOVs, an SFPCOMPC and a store.
        sfpi::dst_reg[0] = 1.0f;
        v_if(v == s) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;

        sfpi::dst_reg++;
    }
}

inline void unary_eq_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_eq(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Default stored first, taken arm stored under the predicate (see unary_ne).
        sfpi::dst_reg[0] = 0.0f;
        v_if(v == s) { sfpi::dst_reg[0] = 1.0f; }
        v_endif;

        sfpi::dst_reg++;
    }
}

inline void unary_gt_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_gt(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Default stored first, taken arm stored under the predicate (see unary_ne).
        sfpi::dst_reg[0] = 0.0f;
        v_if(v > s) { sfpi::dst_reg[0] = 1.0f; }
        v_endif;

        sfpi::dst_reg++;
    }
}

inline void unary_lt_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_lt(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Default stored first, taken arm stored under the predicate (see unary_ne).
        sfpi::dst_reg[0] = 0.0f;
        v_if(v < s) { sfpi::dst_reg[0] = 1.0f; }
        v_endif;

        sfpi::dst_reg++;
    }
}

inline void unary_ge_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_ge(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Default stored first, taken arm stored under the predicate (see unary_ne).
        sfpi::dst_reg[0] = 1.0f;
        v_if(v < s) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;

        sfpi::dst_reg++;
    }
}

inline void unary_le_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_unary_le(uint value) {
    // SFPU microcode
    sfpi::vFloat s = Converter::as_float(value);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Default stored first, taken arm stored under the predicate (see unary_ne).
        sfpi::dst_reg[0] = 1.0f;
        v_if(v > s) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;

        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
