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

#define POLYVAL10(coef10, coef9, coef8, coef7, coef6, coef5, coef4, coef3, coef2, coef1, coef0, t4)               \
    ((coef0 +                                                                                                     \
      (coef1 +                                                                                                    \
       (coef2 +                                                                                                   \
        (coef3 +                                                                                                  \
         (coef4 + (coef5 + (coef6 + (coef7 + (coef8 + (coef9 + coef10 * t4) * t4) * t4) * t4) * t4) * t4) * t4) * \
            t4) *                                                                                                 \
           t4) *                                                                                                  \
          t4) *                                                                                                   \
     t4)
// Park the three costliest series coefficients in the SFPU's programmable constant
// registers: SFPMAD names a CREG directly in an operand field, so each costs zero
// instructions per element instead of the two SFPLOADI a full-fp32 literal needs, and the
// value keeps its exact fp32 bit pattern (bit-exact, not an accuracy trade). The other
// coefficients stay literals -- 0.25f and 0.015625f are already bf16-exact and free, and a
// dependent Horner chain hides some load latency, so draining every load out of the chain
// trades loads for stalls rather than removing them outright.
inline void i0_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 0.0004340277778f;
    sfpi::vConstFloatPrgm1 = 0.000006781684028f;
    sfpi::vConstFloatPrgm2 = 6.78E-08f;
}

// Horner step of the I0 series, applied to two independent accumulators in lockstep.
// The steps must alternate at STATEMENT level: GCC will not interleave two independent
// expression trees on its own. Coefficients ascend exactly as POLYVAL10 consumed them, so
// each element's chain keeps its operations, operands and order: bit-exact.
#define I0_STEP2(c)         \
    do {                    \
        r0 = r0 * t0 + (c); \
        r1 = r1 * t1 + (c); \
    } while (0)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_i0() {
    // Two elements per iteration, hand-interleaved. The series is a fully dependent Horner
    // chain, so every step whose coefficient is a CREG or an immediate stalls on the previous
    // SFPMAD with nothing to fill the slot. Two independent chains fill those slots, and one
    // TTINCRWC covers both elements.
    //
    // Register pressure: 2 x (t, r) plus the two loads = 6 of the 8 general LRegs, with the
    // coefficients in CREGs. A third interleaved element spills.
#pragma GCC unroll 0

    for (int d = 0; d < ITERATIONS / 2; d++) {
        vFloat in0 = dst_reg[0];
        vFloat in1 = dst_reg[1];
        vFloat t0 = in0 * in0;
        vFloat t1 = in1 * in1;

        vFloat r0 = 1.50E-22f;
        vFloat r1 = 1.50E-22f;
        I0_STEP2(7.24E-20f);
        I0_STEP2(2.90E-17f);
        I0_STEP2(9.39E-15f);
        I0_STEP2(2.40E-12f);
        I0_STEP2(4.71E-10f);
        I0_STEP2(sfpi::vConstFloatPrgm2);  // 6.78E-08f,          parked by i0_init
        I0_STEP2(sfpi::vConstFloatPrgm1);  // 0.000006781684028f, parked by i0_init
        I0_STEP2(sfpi::vConstFloatPrgm0);  // 0.0004340277778f,   parked by i0_init
        I0_STEP2(0.015625f);
        I0_STEP2(0.25f);
        r0 = r0 * t0;  // POLYVAL10's trailing multiply by t4
        r1 = r1 * t1;

        dst_reg[0] = 1.0f + r0;
        dst_reg[1] = 1.0f + r1;

        dst_reg += 2;
    }

    // Odd ITERATIONS: finish the trailing element on its own.
    if constexpr (ITERATIONS % 2 != 0) {
        vFloat in = dst_reg[0];
        vFloat t = in * in;
        vFloat r = 1.50E-22f;
        r = r * t + 7.24E-20f;
        r = r * t + 2.90E-17f;
        r = r * t + 9.39E-15f;
        r = r * t + 2.40E-12f;
        r = r * t + 4.71E-10f;
        r = r * t + sfpi::vConstFloatPrgm2;
        r = r * t + sfpi::vConstFloatPrgm1;
        r = r * t + sfpi::vConstFloatPrgm0;
        r = r * t + 0.015625f;
        r = r * t + 0.25f;
        dst_reg[0] = 1.0f + r * t;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
