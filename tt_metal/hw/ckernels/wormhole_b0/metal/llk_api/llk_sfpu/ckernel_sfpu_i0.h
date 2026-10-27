// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // Three mid-chain Horner coefficients live in the programmable constant registers, so
    // the row body reads them as MAD operands instead of re-issuing their SFPLOADI pairs.
    // Nothing else in i0 touches these registers.
    vConstFloatPrgm0 = 0.000006781684028f;
    vConstFloatPrgm1 = 0.0004340277778f;
    vConstFloatPrgm2 = 6.78E-08f;
}

// I0(x) ~= 1 + t*(c0 + t*(c1 + ... + t*c10)), t = x*x: the same degree-10 Horner chain in t
// as before, evaluated for two rows per trip so the second chain's MADs fill the first's
// latency shadow. The three leading coefficients stay in LREGs across the loop; every
// per-row operation is the same SFPMAD on the same fp32 operands in the same order, so
// the result is bit-identical to the one-row form.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    static_assert(ITERATIONS % 2 == 0, "calculate_i0 processes two rows per trip");
    const vFloat c10 = 1.50E-22f;
    const vFloat c9 = 7.24E-20f;
    const vFloat c8 = 2.90E-17f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d += 2) {
        vFloat in0 = dst_reg[0];
        vFloat in1 = dst_reg[1];
        vFloat t0 = in0 * in0;
        vFloat t1 = in1 * in1;
        vFloat r0 = t0 * c10 + c9;
        vFloat r1 = t1 * c10 + c9;
        r0 = r0 * t0 + c8;
        r1 = r1 * t1 + c8;
        vFloat c = 9.39E-15f;
        r0 = r0 * t0 + c;
        r1 = r1 * t1 + c;
        c = 2.40E-12f;
        r0 = r0 * t0 + c;
        r1 = r1 * t1 + c;
        c = 4.71E-10f;
        r0 = r0 * t0 + c;
        r1 = r1 * t1 + c;
        r0 = r0 * t0 + vConstFloatPrgm2;
        r1 = r1 * t1 + vConstFloatPrgm2;
        r0 = r0 * t0 + vConstFloatPrgm0;
        r1 = r1 * t1 + vConstFloatPrgm0;
        r0 = r0 * t0 + vConstFloatPrgm1;
        r1 = r1 * t1 + vConstFloatPrgm1;
        c = 0.015625f;
        r0 = r0 * t0 + c;
        r1 = r1 * t1 + c;
        c = 0.25f;
        r0 = r0 * t0 + c;
        r1 = r1 * t1 + c;
        dst_reg[0] = r0 * t0 + 1.0f;
        dst_reg[1] = r1 * t1 + 1.0f;
        dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
