// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Sets p (central, u >= -7: every bf16/fp16 input) and t (fp32 tail, u down to -16) to erfinv(x) / x,
// u = ln(1 - x^2) (Giles 2010); the caller multiplies by x, selects t where u_plus_7 < 0, copies x's sign.
template <bool APPROXIMATION_MODE>
sfpi_inline void calculate_erfinv_body(sfpi::vFloat x, sfpi::vFloat& p, sfpi::vFloat& t, sfpi::vFloat& u_plus_7) {
    sfpi::vFloat u = calculate_log_body<false, false, true, false, true>(1.0f - x * x, 0);

    // Central polynomial, u in [-7, 0], degree 7.
    p = -0x1.3p-22f;
    p = p * u + -0x1.0ap-17f;
    p = p * u + -0x1.47cp-14f;
    p = p * u + -0x1.6b4p-13f;
    p = p * u + 0x1.3p-9f;
    p = p * u + 0x1.7b4fap-7f;
    p = p * u + -0x1.db1b3ap-3f;
    p = p * u + 0x1.c5c02p-1f;

    // Computed ahead of the tail polynomial so the caller's SFPSETCC doesn't wait on this SFPMAD.
    u_plus_7 = u + 7.0f;

    // Tail polynomial, u in [-16, -7], degree 4.
    t = -0x1.aep-18f;
    t = t * u + -0x1.c24p-12f;
    t = t * u + -0x1.a58p-7f;
    t = t * u + -0x1.405c58p-2f;
    t = t * u + 0x1.93939cp-1f;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() {
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat p, t, u_plus_7;
        calculate_erfinv_body<false>(sfpi::dst_reg[0], p, t, u_plus_7);
        // x is reloaded rather than kept live across the log, which needs every LREG.
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Selecting after the multiply puts the SFPSETCC in the SFPMUL's result-latency slot.
        sfpi::vFloat r = p * x;
        v_if(u_plus_7 < 0.0f) { r = t * x; }
        v_endif;
        sfpi::dst_reg[0] = sfpi::copysgn(r, x);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erfinv_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    log_init<false, false, true>();
}

}  // namespace sfpu
}  // namespace ckernel
