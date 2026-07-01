// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_expm1.h"
#include "ckernel_sfpu_trigonometry.h"

namespace ckernel::sfpu {

// tanh(x): t = 0.5*expm1(abs(2*x)); sgn(x) * t / (t + 1)
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat x) {
    sfpi::vFloat a, r, s, f, w, y, scale, bias0, bias1, c0;
    sfpi::vFloat t, rcp, x0, x1, tmp, y0;
    sfpi::vInt x_exp;

    sfpi::vFloat j = x * sfpi::vConstFloatPrgm0;  // j = x * 2 * log2(e)
    a = x + x;
    // Rounds the absolute value of j, clamped to [0, 255].
    sfpi::vMag m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);
    sfpi::vInt i = m;

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

    sfpi::vInt e = i + 126;
    r = r * s + f;
    scale = sfpi::setexp(sfpi::vConst0, e);
    bias0 = scale - w;
    a = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - 1);
    x0 = r * scale + bias0;
    y = a * 0.0f + 1.0f;
    x1 = x0 + 1.0f;

    v_if(i < 61) {
        // computes x0/x1 via reciprocal and residual correction
        rcp = sfpi::approx_recip(x1);
        t = -x1 * rcp + 1.0f;
        y = x;
        rcp = rcp * t + rcp;
        y0 = x0 * rcp;
        x_exp = sfpi::exexp(x, sfpi::ExponentMode::NoDebias);
        t = -x1 * y0 + x0;
        v_if(x_exp >= 115) { y = t * rcp + y0; }
        v_endif;
    }
    v_endif;

    return sfpi::copysgn(y, x);
}

template <bool is_fp32_acc_to_dest_mode>
sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    // For negative numbers, we compute tanh(-x) = -tanh(x)
    sfpi::vFloat val = sfpi::abs(x);  // set positive

    // Polynomial coefficients found using Sollya
    // val * (0.999004364013671875 + val * (3.0897438526153564453125e-2 + val * (-0.4890659749507904052734375 + val *
    // (0.281917631626129150390625 + val * (-6.6649019718170166015625e-2 + val *
    // (5.876733921468257904052734375e-3))))));
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        sfpi::vConst0,
        0.999004364013671875,
        3.0897438526153564453125e-2,
        -0.4890659749507904052734375,
        sfpi::vConstFloatPrgm2,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0);

    // For larger x, the polynomial approximation may exceed 1.0.
    // Since tanh(x) is bounded by [-1, 1], we clamp output to 1.0.
    sfpi::vFloat threshold_value = sfpi::vConst1;
    sfpi::vec_min_max(result, threshold_value);

    result = sfpi::copysgn(result, x);  // restore sign (i.e. tanh(-x) = -tanh(x))

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tanh() {
    if constexpr (APPROXIMATION_MODE) {
        // SFPU microcode
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
    } else {  // APPROXIMATION_MODE is false

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
