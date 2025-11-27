// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool floor>
inline void calculate_div_int32_body(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    sfpi::vInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
    sfpi::vInt sign = a ^ b;
    a = sfpi::abs(a);
    b = sfpi::abs(b);

    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);
    v_if(a_f < 0.0f) { a_f = 2147483648.0f; }
    v_endif;
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(b_f < 0.0f) { b_f = 2147483648.0f; }
    v_endif;

    sfpi::vFloat inv_b_f = _sfpu_reciprocal_<2>(b_f);  // accurate to ~23 bits

    // initial approximation q = a * 1/b
    sfpi::vFloat q_f = a_f * inv_b_f;
    sfpi::vInt q = 0;
    sfpi::vInt exp = sfpi::exexp(q_f);
    v_if(exp >= 0) {
        q = sfpi::exman8(q_f);
        exp = exp - 23;
        q = q << exp;
    }
    v_endif;

    // compute qb = q * b

    sfpi::vInt q1 = q >> 23;
    sfpi::vInt b1 = b >> 23;
    sfpi::vInt lo;
    sfpi::vInt hi;

    sfpi::l_reg[LRegs::LReg1] = q1;
    sfpi::l_reg[LRegs::LReg2] = b;
    TTI_SFPMUL24(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // q1 = b*q1
    q1 = sfpi::l_reg[LRegs::LReg1];

    sfpi::l_reg[LRegs::LReg0] = q;
    sfpi::l_reg[LRegs::LReg3] = b1;
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);  // b1 = q*b1
    b1 = sfpi::l_reg[LRegs::LReg3];
    q1 += b1;

    sfpi::l_reg[LRegs::LReg0] = q;
    sfpi::l_reg[LRegs::LReg1] = b;
    sfpi::l_reg[LRegs::LReg2] = lo;
    sfpi::l_reg[LRegs::LReg3] = hi;
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);  // lo = q*b
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG3, 1);  // hi = q*b
    lo = sfpi::l_reg[LRegs::LReg2];
    hi = sfpi::l_reg[LRegs::LReg3];
    q1 += hi;

    lo += q1 << 23;
    sfpi::vInt r = a - lo;
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r));
    sfpi::vFloat correction_f = r_f * inv_b_f;

    sfpi::vInt correction = 0;
    exp = sfpi::exexp(correction_f);
    v_if(exp >= 0) {
        correction = sfpi::exman8(correction_f);
        exp = exp - 23;
        correction = correction << exp;
    }
    v_endif;
    v_if(r < 0) { correction = ~correction; }
    v_endif;
    q += correction;

    v_if(sign < 0) {
        q = -q;
        v_if(r != 0) {
            if constexpr (floor) {
                q += 1;
            } else {
                q -= 1;
            }
        }
        v_endif;
    }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = q;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32_floor(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_div_int32_body<true>(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32_trunc(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_div_int32_body<false>(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void div_floor_init() {
    _init_sfpu_reciprocal_<false>();
}

template <bool APPROXIMATION_MODE>
inline void div_trunc_init() {
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
