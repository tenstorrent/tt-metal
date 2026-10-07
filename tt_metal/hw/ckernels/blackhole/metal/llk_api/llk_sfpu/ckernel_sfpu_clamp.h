// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

enum { Max = true, Min = false };  // Clamp Mode

inline void clamp_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// out = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint min_val, uint max_val) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(min_val);
        calculate_unary_max_min_float_body<Max>();
        load_value_param_float(max_val);
        calculate_unary_max_min_float_body<Min>();
        sfpi::dst_reg++;
    }
}

// One pass with the bounds in L12 and L13; a swap runs in the inverted domain when its bound is negative, as in
// calculate_unary_max_min_int32_body.
template <bool MIN_NEG, bool MAX_NEG, int ITERATIONS>
inline void clamp_int32_rows() {
    sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        if constexpr (MIN_NEG) {
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        }
        TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, MIN_NEG ? sfpi::SFPSWAP_MOD1_VEC_MIN_MAX : 9);
        if constexpr (MIN_NEG != MAX_NEG) {
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        }
        TTI_SFPSWAP(0, p_sfpu::LREG13, p_sfpu::LREG0, MAX_NEG ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        if constexpr (MAX_NEG) {
            TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        }
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp_int32(uint min_val, uint max_val) {
    const bool min_neg = static_cast<int>(min_val) < 0;
    const bool max_neg = static_cast<int>(max_val) < 0;
    load_value_param_int(min_val);
    sfpi::vConstIntPrgm1 = max_neg ? ~max_val : max_val;
    if (min_neg) {
        if (max_neg) {
            clamp_int32_rows<true, true, ITERATIONS>();
        } else {
            clamp_int32_rows<true, false, ITERATIONS>();
        }
    } else if (max_neg) {
        clamp_int32_rows<false, true, ITERATIONS>();
    } else {
        clamp_int32_rows<false, false, ITERATIONS>();
    }
}

}  // namespace ckernel::sfpu
