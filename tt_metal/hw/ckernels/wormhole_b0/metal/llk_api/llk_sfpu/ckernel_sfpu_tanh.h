// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_expm1.h"
#include "ckernel_sfpu_trigonometry.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// tanh(x): t = 0.5*expm1(abs(2*x)); sgn(x) * t / (t + 1)
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat x) {
    sfpi::vFloat a, r, s, f, w, y, scale, bias0;
    sfpi::vFloat j, t, rcp, x0, x1, y0;
    sfpi::vInt i, magic_seed, e, x_exp;
    sfpi::vMag m;

    // Calculate j = x * (2 * log2(e)), interleaved with a = abs(2*x), and i = round(abs(j)), clamped to [0, 255].

    j = x * sfpi::vConstFloatPrgm0;  // j = x * 2 * log2(e)
    a = x + x;
    // i = round(abs(j)), clamped to [0, 255].
    m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    i = m;
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);

    a = sfpi::setsgn(a, 0);
    f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

    // expm1(f)
    r = 1.974105835e-04f;
    r = r * f + 1.393318176e-3f;
    r = r * f + 8.331298828e-3f;
    r = r * f + 4.166680202e-2f;
    s = f * f;  // hide SFPMAD latency
    r = r * f + sfpi::vConstFloatPrgm2;
    w = 0.5f;
    r = __builtin_rvtt_sfpmad(r.get(), f.get(), w.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    e = i + 126;
    r = r * s + f;
    scale = sfpi::setexp(0.0f, e);
    bias0 = scale - w;

    // If a=±inf, converts to a finite value, otherwise if a=±NaN, converts to ±inf or ±NaN.
    // This gives y = <finite value> * 0.0 + 1.0 = 1.0 for non-NaN x, otherwise y = NaN.
    a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - 1);
    x0 = r * scale + bias0;
    y = a * 0.0f + 1.0f;
    x1 = x0 + 1.0f;

    // computes x0/x1 via reciprocal and residual correction
    magic_seed = 0xfef30000;
    rcp = sfpi::as<sfpi::vFloat>(magic_seed - sfpi::as<sfpi::vInt>(x1));
    t = x1 * rcp + 1.0f;

    // `i` is round(abs(2*x/log(2))). For i >= 61, |x| is about 21 or larger,
    // so x0/(x0 + 1) is far within 0.5 ulp of 1.0f. Keep the preinitialized
    // saturated result; below that, refine the negative reciprocal estimate.
    v_if(i < 61) {
        t = t * t + t;
        y = x;
        rcp = rcp * t + rcp;
        x_exp = sfpi::exexp(x, sfpi::ExponentMode::Biased);
        y0 = x0 * rcp;
        t = x1 * y0 + x0;

        // For tiny inputs, tanh(x) rounds to x in fp32. `x_exp` is biased, so
        // 115 is 127 - 12; keep y=x for |x| < 2^-12 and use the corrected
        // ratio otherwise.
        v_if(x_exp >= 115) { y = t * rcp + y0; }
        v_endif;
    }
    v_endif;

    return sfpi::copysgn(y, x);
}

// Sollya coefficients. tanh_init has programmable CRegs for the top three only, so these three
// cost an SFPLOADI pair per use unless the caller keeps them in an LReg.
// val * (0.999004364013671875 + val * (3.0897438526153564453125e-2 + val * (-0.4890659749507904052734375 + val *
// (0.281917631626129150390625 + val * (-6.6649019718170166015625e-2 + val *
// (5.876733921468257904052734375e-3))))));
constexpr float TANH_POLY_C1 = 0.999004364013671875f;
constexpr float TANH_POLY_C2 = 3.0897438526153564453125e-2f;
constexpr float TANH_POLY_C3 = -0.4890659749507904052734375f;

sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    // For negative numbers, we compute tanh(-x) = -tanh(x)
    sfpi::vFloat val = sfpi::abs(x);  // set positive

    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        0.0f,
        TANH_POLY_C1,
        TANH_POLY_C2,
        TANH_POLY_C3,
        sfpi::vConstFloatPrgm2,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0);

    // For larger x, the polynomial approximation may exceed 1.0.
    // Since tanh(x) is bounded by [-1, 1], we clamp output to 1.0.
    result = sfpi::min(result, 1.0f);

    result = sfpi::copysgn(result, x);  // restore sign (i.e. tanh(-x) = -tanh(x))

    return result;
}

// Two datums through the polynomial in lockstep, so each fills the other's SFPMAD stall slots.
// Only WH stalls; BH comes out even either way, so both arches run this shape. Only c1 can be
// hoisted on top of it: six vectors are already live for the data, and an eighth spills.
sfpi_inline void _sfpu_tanh_polynomial_x2_(
    sfpi::vFloat& y0, sfpi::vFloat& y1, sfpi::vFloat x0, sfpi::vFloat x1, sfpi::vFloat c1) {
    sfpi::vFloat a0 = sfpi::abs(x0);
    sfpi::vFloat a1 = sfpi::abs(x1);

    sfpi::vFloat r0 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat r1 = sfpi::vConstFloatPrgm0;
    r0 = r0 * a0 + sfpi::vConstFloatPrgm1;
    r1 = r1 * a1 + sfpi::vConstFloatPrgm1;
    r0 = r0 * a0 + sfpi::vConstFloatPrgm2;
    r1 = r1 * a1 + sfpi::vConstFloatPrgm2;
    // One local each, else sfpi emits the SFPLOADI pair per MAD. Both die after their second use.
    sfpi::vFloat c3 = TANH_POLY_C3;
    r0 = r0 * a0 + c3;
    r1 = r1 * a1 + c3;
    sfpi::vFloat c2 = TANH_POLY_C2;
    r0 = r0 * a0 + c2;
    r1 = r1 * a1 + c2;
    r0 = r0 * a0 + c1;
    r1 = r1 * a1 + c1;
    r0 = r0 * a0;
    r1 = r1 * a1;

    y0 = sfpi::copysgn(sfpi::min(r0, 1.0f), x0);
    y1 = sfpi::copysgn(sfpi::min(r1, 1.0f), x1);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        // 6-segment SFPLUTFP32, replacing the legacy 3-entry SFPLUT. See tanh_init.
        sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];
        sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];
        sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];
        sfpi::vUInt l4 = l_reg[sfpi::LRegs::LReg4];
        sfpi::vUInt l5 = l_reg[sfpi::LRegs::LReg5];
        sfpi::vUInt l6 = l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = sfpi::copysgn(lut2_sign(val, l0, l1, l2, l4, l5, l6), val);

            sfpi::dst_reg++;
        }

        l_reg[sfpi::LRegs::LReg0] = l0;
        l_reg[sfpi::LRegs::LReg1] = l1;
        l_reg[sfpi::LRegs::LReg2] = l2;
        l_reg[sfpi::LRegs::LReg4] = l4;
        l_reg[sfpi::LRegs::LReg5] = l5;
        l_reg[sfpi::LRegs::LReg6] = l6;
    } else if constexpr (is_fp32_dest_acc_en) {  // APPROXIMATION_MODE is false
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_tanh_fp32_accurate_(val);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        sfpi::vFloat c1 = TANH_POLY_C1;  // inline it and every datum pays an SFPLOADI pair

        // Walk dst_reg rather than index by d: a uniform body is what the replay buffer records
        // once, and a runtime index makes sfpi build each SFPLOAD/SFPSTORE in scalar registers.
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat r0, r1;
            _sfpu_tanh_polynomial_x2_(r0, r1, sfpi::dst_reg[0], sfpi::dst_reg[1], c1);
            // Round into a vFloat; storing the vFloat16b expression pins SFPSTORE to FP16B.
            r0 = sfpi::convert<sfpi::vFloat16b>(r0, sfpi::RoundMode::Nearest);
            r1 = sfpi::convert<sfpi::vFloat16b>(r1, sfpi::RoundMode::Nearest);

            sfpi::dst_reg[0] = r0;
            sfpi::dst_reg[1] = r1;
            sfpi::dst_reg += 2;
        }

        if constexpr (ITERATIONS % 2 != 0) {
            sfpi::vFloat result = _sfpu_tanh_polynomial_(sfpi::dst_reg[0]);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void tanh_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        // 6-segment piecewise linear via SFPLUTFP32, replacing the legacy 3-entry
        // SFPLUT (slopes 0.90625 / 0.09375 then a hard 1.0, breakpoints at 1.0 and
        // 2.0 only, 8-bit coefficients). That table measures 0.145 max abs error on
        // device; this one measures ~0.0125 by the same sweep -- ~12x better for one
        // extra cycle, since SFPLUTFP32 is 2 cycles against SFPLUT's 1 and the
        // instruction count is unchanged.
        //
        // Hardware breakpoints on |x| (FP16 6-entry TABLE1, sfpi mode = 1):
        //   [0.0, 0.5): 0.949219*|x|
        //   [0.5, 1.0): 0.599121*|x| + 0.174194
        //   [1.0, 1.5): 0.287109*|x| + 0.481934
        //   [1.5, 2.0): 0.117737*|x| + 0.731934
        //   [2.0, 3.0): 0.030975*|x| + 0.905762
        //   [3.0, inf): 1.0
        //
        // Segment 0's intercept is pinned to 0 so tanh(0) == 0 and the fit is
        // continuous across the sign flip. Slopes pack lo/hi into LReg0/1/2 and
        // intercepts into LReg4/5/6; a half-word of 0x7C00 reads as 0.0. The result
        // is signed with copysgn because sfpi 7.71.0 only admits the SGN_UPDATE
        // form of the 6-entry LUT (see ckernel_sfpu_erf.h).
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x38CB3B98);
        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x2F893498);
        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x7C0027EE);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0x31937C00);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0x39DB37B6);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vUInt(0x3C003B3F);
    } else {
        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vConstFloatPrgm0 = 2.0f * 1.442695f;      // 2 * log2(e) == 2 / ln(2)
            sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // ln(2)
            sfpi::vConstFloatPrgm2 = 1.666667163e-1f;       // c1
        } else {
            // Polynomial approximation
            // Store some polynomial coefficients in programmable registers
            sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
            sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
            sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
        }
    }
}

}  // namespace ckernel::sfpu
