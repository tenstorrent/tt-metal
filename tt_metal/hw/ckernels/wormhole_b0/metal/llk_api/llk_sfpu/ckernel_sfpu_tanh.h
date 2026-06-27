// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Yugansh Tyagi
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_trigonometry.h"

namespace ckernel::sfpu {

/*
 * tanh(x) for fp32 via expm1 with Estrin-scheme polynomial evaluation.
 *
 * Mathematical identity:
 *   tanh(x) = expm1(2|x|) / (expm1(2|x|) + 2),  result gets sign(x)
 *
 * Algorithm:
 *   1. Range reduce: a = |2x|, i = rint(a/ln2) in [0,255], f = a - i*ln2
 *   2. expm1(f) via degree-5 minimax using Estrin evaluation scheme
 *   3. Reconstruct via setexp + divide via approx_recip + Newton-Raphson
 *   4. Saturate to +/-1 when i >= 61
 *
 * Key optimisation over sigmoid-split approach (#48287):
 *   The degree-5 minimax polynomial is evaluated using Estrin's scheme
 *   instead of Horner's.  Estrin groups coefficients into three independent
 *   pairs (q_lo, q_mi, q_hi) that share no data dependencies on each other,
 *   allowing the SFPU to issue them back-to-back without stall bubbles.
 *
 *   Horner critical path:  ~8 sequential MAD latencies
 *   Estrin critical path:  ~4 MAD latencies (3 levels + 1 final)
 *
 *   Expected cycle saving: ~6-8 cycles on the polynomial evaluation alone.
 *
 * Polynomial coefficients (degree-5 minimax, f in [-ln2/2, ln2/2]):
 *   c0 = 0.5
 *   c1 = 1/6                 (vConstFloatPrgm2)
 *   c2 = 4.166680202e-2
 *   c3 = 8.331298828e-3
 *   c4 = 1.393318176e-3
 *   c5 = 1.974105835e-4
 *
 * Target: < 40 cycles (WH), maxulperr < 3.
 */
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat x) {
    sfpi::vFloat a, r, f, x0, x1;
    sfpi::vFloat t, rcp, y0, y;
    sfpi::vInt x_exp;

    // Step 1: range reduction
    //   j = rint(|2x| / ln2)  clamped to uint8 [0, 255]
    //   f = |2x| - j * ln2    in [-ln2/2, ln2/2]
    sfpi::vFloat j = x * sfpi::vConstFloatPrgm0;   // j = x * 2*log2(e)
    a = x + x;
    sfpi::vMag m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);
    sfpi::vInt i = m;

    a = sfpi::setsgn(a, 0);                         // a = |2x|
    f = j * sfpi::vConstFloatPrgm1 + a;             // f = |2x| - j*ln2

    // Step 2: expm1(f) via Estrin-scheme degree-5 minimax polynomial
    //
    //   expm1(f) = f + f^2 * q(f)
    //   q(f) = c0 + c1*f + c2*f^2 + c3*f^3 + c4*f^4 + c5*f^5
    //
    //   Estrin factoring:
    //   q(f) = (c0+c1*f) + f^2 * ((c2+c3*f) + f^2*(c4+c5*f))
    //
    //   The four Level-1 ops below (f2, q_lo, q_mi, q_hi) all depend only
    //   on f and are mutually independent.  The SFPU can issue them in
    //   consecutive cycles with no inter-instruction stalls, pipelining
    //   the latency of each behind the others.
    sfpi::vFloat f2    = f * f;                                    // Level 1
    sfpi::vFloat q_lo  = sfpi::vConstFloatPrgm2 * f + 0.5f;       // Level 1: c1*f + c0
    sfpi::vFloat q_mi  = 8.331298828e-3f * f + 4.166680202e-2f;   // Level 1: c3*f + c2
    sfpi::vFloat q_hi  = 1.974105835e-04f * f + 1.393318176e-3f;  // Level 1: c5*f + c4

    sfpi::vFloat q_mid = f2 * q_hi + q_mi;                        // Level 2
    sfpi::vFloat q     = f2 * q_mid + q_lo;                       // Level 3
    r                  = f2 * q + f;                               // expm1(f)

    // Step 3: reconstruct expm1(|2x|) from expm1(f) and 2^i
    //   scale = 2^(i-1),  bias0 = scale - 0.5
    //   x0    = 0.5 * expm1(|2x|)
    //   x1    = 0.5 * (expm1(|2x|) + 2)  =  x0 + 1
    sfpi::vInt e       = i + 126;
    sfpi::vFloat scale = sfpi::setexp(sfpi::vConst0, e);
    sfpi::vFloat w     = 0.5f;
    sfpi::vFloat bias0 = scale - w;
    // Decrement a's bits by 1 so that a*0 = NaN when a = NaN (IEEE-754 safe)
    a  = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - 1);
    x0 = r * scale + bias0;
    y  = a * 0.0f + 1.0f;   // y = 1.0, NaN-propagating
    x1 = x0 + 1.0f;

    // Step 4: divide x0/x1 via reciprocal + Newton-Raphson + residual correction
    //   For i >= 61 (|x| > ~21), tanh = +/-1 exactly in fp32, skip division.
    v_if(i < 61) {
        rcp   = sfpi::approx_recip(x1);        // ~12-bit initial 1/x1
        t     = -x1 * rcp + 1.0f;             // N-R error term
        y     = x;                              // fallback: tanh(x) ~= x for tiny |x|
        rcp   = rcp * t + rcp;                 // ~24-bit after one N-R step
        y0    = x0 * rcp;                      // first quotient estimate
        x_exp = sfpi::exexp(x, sfpi::ExponentMode::NoDebias);
        t     = -x1 * y0 + x0;                // residual
        v_if(x_exp >= 115) {                   // apply correction for |x| >= 2^{-12}
            y = t * rcp + y0;
        }
        v_endif;
    }
    v_endif;

    return sfpi::copysgn(y, x);
}

template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    sfpi::vFloat val = sfpi::abs(x);

    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        sfpi::vConst0,
        0.999004364013671875,
        3.0897438526153564453125e-2,
        -0.4890659749507904052734375,
        sfpi::vConstFloatPrgm2,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0);

    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::copysgn(result, x);
    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            val = sfpi::lut(val, l0, l1, l2);
            sfpi::dst_reg[0] = val;
            sfpi::dst_reg++;
        }

        l_reg[sfpi::LRegs::LReg0] = l0;
        l_reg[sfpi::LRegs::LReg1] = l1;
        l_reg[sfpi::LRegs::LReg2] = l2;
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];

            sfpi::vFloat result;

            if constexpr (is_fp32_dest_acc_en) {
                result = _sfpu_tanh_fp32_accurate_(val);
            } else {
                result = _sfpu_tanh_polynomial_<is_fp32_dest_acc_en>(val);
                result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init() {
    if constexpr (APPROXIMATION_MODE) {
        uint imm0 = 0x1DFF;  // 0.90625*x
        uint imm1 = 0x481A;  // 0.09375*x + 0.8125
        uint imm2 = 0xFF00;  // 1
        _sfpu_load_imm16_(0, imm0);
        _sfpu_load_imm16_(1, imm1);
        _sfpu_load_imm16_(2, imm2);
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            // prgm0 = 2*log2(e)  — range-reduction multiplier
            // prgm1 = -ln(2)     — range-reduction subtractor
            // prgm2 = 1/6        — Estrin level-1 coefficient c1
            sfpi::vConstFloatPrgm0 = 2.0f * 1.442695f;      // 2 * log2(e)
            sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // -ln(2)
            sfpi::vConstFloatPrgm2 = 1.666667163e-1f;       // 1/6
        } else {
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
        }
    }
}

}  // namespace ckernel::sfpu
