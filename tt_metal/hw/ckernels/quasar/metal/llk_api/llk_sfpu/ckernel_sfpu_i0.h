// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Series I0(x) = sum_k A<k> * t^k, t = x*x, A<k> = 1 / (4^k * (k!)^2), truncated after k = 11.
struct I0Series {
    static constexpr float A0 = 1.0f;
    static constexpr float A1 = 0.25f;
    static constexpr float A2 = 0.015625f;
    static constexpr float A3 = 0.0004340277778f;
    static constexpr float A4 = 0.000006781684028f;
    static constexpr float A5 = 6.78E-08f;
    static constexpr float A6 = 4.71E-10f;
    static constexpr float A7 = 2.40E-12f;
    static constexpr float A8 = 9.39E-15f;
    static constexpr float A9 = 2.90E-17f;
    static constexpr float A10 = 7.24E-20f;
    static constexpr float A11 = 1.50E-22f;
};

/**
 * @brief Programs the state calculate_i0 reads: vConstFloatPrgm0-2 = I0Series::A5, A4, A3, and
 *        ADDR_MOD_6 = {dest.incr = 4}, two sfpi rows, which the paired SFPSTORE selects directly
 *        (the Quasar SFPSTORE carries a 3-bit addr-mod index). The SFPU common init owns ADDR_MOD_7
 *        and the matmul ADDR_MOD_0..5, so ADDR_MOD_6 is free for the op, as for logaddexp.
 * @note Run it after any other SFPU op's init and before calculate_i0: both the programmable
 *       constant registers and ADDR_MOD_6 are shared with other ops' inits.
 */
inline void i0_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2 * SFP_DESTREG_STRIDE}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    vConstFloatPrgm0 = I0Series::A5;
    vConstFloatPrgm1 = I0Series::A4;
    vConstFloatPrgm2 = I0Series::A3;
}

/**
 * @brief I0(x) for ITERATIONS dest rows: the Horner chain of I0Series in t = x*x, evaluated for
 *        two rows per trip so the second row's SFPMADs fill the first row's dependency stalls.
 *        A11..A9 stay in LREGs across the loop, A5..A3 come from vConstFloatPrgm0-2.
 * @tparam ITERATIONS: Dest rows to process; must be even.
 * @note Call i0_init first; this reads vConstFloatPrgm0-2 and ADDR_MOD_6 as it programs them.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    static_assert(ITERATIONS % 2 == 0, "calculate_i0 processes two rows per trip");
    const vFloat a11 = I0Series::A11;
    const vFloat a10 = I0Series::A10;
    const vFloat a9 = I0Series::A9;

    // Not unrolled: sfpi rolls an unrolled body into the replay buffer from slot 0, and this
    // body (38 instructions per trip) would not fit the replay window the FPU shares.
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d += 2) {
        vFloat in0 = dst_reg[0];
        vFloat in1 = dst_reg[1];
        vFloat t0 = in0 * in0;
        vFloat t1 = in1 * in1;
        vFloat r0 = t0 * a11 + a10;
        vFloat r1 = t1 * a11 + a10;
        r0 = r0 * t0 + a9;
        r1 = r1 * t1 + a9;
        vFloat a = I0Series::A8;
        r0 = r0 * t0 + a;
        r1 = r1 * t1 + a;
        a = I0Series::A7;
        r0 = r0 * t0 + a;
        r1 = r1 * t1 + a;
        a = I0Series::A6;
        r0 = r0 * t0 + a;
        r1 = r1 * t1 + a;
        r0 = r0 * t0 + vConstFloatPrgm0;  // A5
        r1 = r1 * t1 + vConstFloatPrgm0;
        r0 = r0 * t0 + vConstFloatPrgm1;  // A4
        r1 = r1 * t1 + vConstFloatPrgm1;
        r0 = r0 * t0 + vConstFloatPrgm2;  // A3
        r1 = r1 * t1 + vConstFloatPrgm2;
        a = I0Series::A2;
        r0 = r0 * t0 + a;
        r1 = r1 * t1 + a;
        a = I0Series::A1;
        r0 = r0 * t0 + a;
        r1 = r1 * t1 + a;
        dst_reg[0] = r0 * t0 + I0Series::A0;
        // ADDR_MOD_6 (i0_init) advances dest by the two rows on this store, in place of a TTINCRWC.
        dst_reg[1].mode(ADDR_MOD_6) = r1 * t1 + I0Series::A0;
    }
}

}  // namespace sfpu
}  // namespace ckernel
