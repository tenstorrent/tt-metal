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

// Approximate tanh: six-segment piecewise-linear table evaluated by SFPLUTFP32
// (FP16_6ENTRY_TABLE1 | SGN_RETAIN), replacing a three-segment SFPLUT table whose
// 4-bit-mantissa coefficients gave max |absolute error| 0.1447. Now 0.0117, at 3 issue slots per
// datum instead of 5.
//
// Raw TTI because sfpi cannot emit the instruction: tanh is odd, so the table is fitted on |x|
// and SGN_RETAIN copies the input's sign onto the result, but sfpi::lut2()'s six-register
// overload always ORs SGN_RETAIN into the mod and __builtin_rvtt_sfplutfp32_6r rejects every mod
// with that bit set, so it can never compile. (lut2_sign() does compile, but SGN_UPDATE would
// give |tanh(x)| and cost a copysign per datum.) The restriction is sfpi's, not the hardware's:
// the mod assembles and is verified on silicon.
//
// The second reason is scheduling. SFPLUTFP32's VD is a free field, but sfpi always picks
// LReg[3] -- the register the instruction is hardwired to read -- so the one-cycle
// result-latency slot has to be an SFPNOP and pipelining through sfpi costs an SFPMOV pair per
// datum. Writing the result to LReg[7] instead lets the next datum's SFPLOAD fill that slot, so
// the steady state is LUT / LOAD-next / STORE-previous and one staging register is enough -- no
// rotation, which is why the table's appetite for LReg[0..2] and LReg[4..6] costs nothing.
constexpr int TANH_APPX_LUT6_MOD = SFPLUTFP32_MOD0_FP16_6ENTRY_TABLE1 | SFPLUTFP32_MOD0_SGN_RETAIN;

template <int K, int ITERATIONS>
sfpi_inline void _tanh_appx_lut6_step_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    // LReg[7] = copysign(table(|LReg[3]|), LReg[3]).
    TTI_SFPLUTFP32(p_sfpu::LREG7, TANH_APPX_LUT6_MOD);

    // Fills the LUT's one-cycle result-latency slot, and is genuinely independent: it writes
    // LReg[3], which the LUT already consumed on issue, and leaves LReg[7] alone.
    if constexpr (K + 1 < ITERATIONS) {
        TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, 2 * (K + 1));
    } else {
        TTI_SFPNOP;
    }

    // Two cycles after the LUT, satisfying its "do not read the result on the next cycle" rule.
    TTI_SFPSTORE(p_sfpu::LREG7, IM, ADDR_MOD_3, 2 * K);

    if constexpr (K + 1 < ITERATIONS) {
        _tanh_appx_lut6_step_<K + 1, ITERATIONS>();
    }
}

template <int ITERATIONS>
inline void _calculate_tanh_appx_lut6_() {
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    // Prologue load; every later load is issued inside the previous datum's latency slot.
    TTI_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_3, 0);
    _tanh_appx_lut6_step_<0, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_tanh_appx_lut6_<ITERATIONS>();
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
        // Six-segment piecewise-linear fit of tanh(|x|), evaluated by SFPLUTFP32 in
        // FP16_6ENTRY_TABLE1 mode: LReg[0..2] hold the slopes, LReg[4..6] the intercepts, two
        // Lut16ToFp32-encoded halves per register (low half = even segment, high half = odd).
        //
        //   |x| <  0.5   0.942871094*|x|                 (intercept pinned to exactly 0)
        //   |x| <  1.0   0.599121094*|x| + 0.174194336
        //   |x| <  1.5   0.287109375*|x| + 0.481933594
        //   |x| <  2.0   0.117736816*|x| + 0.731933594
        //   |x| <  3.0   0.030960083*|x| + 0.905761719
        //   |x| >= 3.0                     1.0
        //
        // Each segment is a minimax linear fit snapped to the nearest Lut16ToFp32-representable
        // pair. Max |absolute error| 0.0117 (was 0.1447), max relative error 0.0571 (was 0.1899).
        // The residual is dominated by the [0.5, 1.0) segment and is the floor for a linear fit
        // on hardware-fixed 0.5-wide breakpoints: err ~ |tanh''|*h^2/16.
        //
        // Both pinned coefficients are load-bearing. The first intercept must stay exactly 0
        // (Lut16ToFp32 code 0x7C00, which encodes zero as exponent 31, not 0x0000) or SGN_RETAIN
        // turns it into +/-c and tanh(0) stops being 0; the tail slope must stay exactly 0 or the
        // fit diverges as |x| grows. A zero tail slope also evaluates 0*inf for |x| = inf, so
        // tanh(+/-inf) is NaN -- unchanged from the old table, which had one too.

        // A0 = 0.942871094 (0x3B8B), A1 = 0.599121094 (0x38CB)
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vUInt(0x38CB3B8B);
        // B0 = 0           (0x7C00), B1 = 0.174194336 (0x3193)
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vUInt(0x31937C00);
        // A2 = 0.287109375 (0x3498), A3 = 0.117736816 (0x2F89)
        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vUInt(0x2F893498);
        // B2 = 0.481933594 (0x37B6), B3 = 0.731933594 (0x39DB)
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vUInt(0x39DB37B6);
        // A4 = 0.030960083 (0x27ED), A5 = 0           (0x7C00)
        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vUInt(0x7C0027ED);
        // B4 = 0.905761719 (0x3B3F), B5 = 1.0         (0x3C00)
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
