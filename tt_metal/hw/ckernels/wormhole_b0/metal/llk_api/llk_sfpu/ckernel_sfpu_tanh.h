// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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

// computes expm1(abs(x))
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_expm1_abs_(sfpi::vFloat x) {
    return y;
}

/*
 * Accurate tanh for fp32 using sigmoid: tanh(x) = 2*sigmoid(2x) - 1
 * For small |x| < 0.6, uses minimax polynomial for better accuracy
 *
 * Algorithm:
 * - For |x| < 0.6: Use minimax polynomial (Sollya-optimized)
 * - For |x| >= 0.6: Use 2*sigmoid(2x) - 1
 *
 * Target accuracy: < 5 ULP for float32 (0.5 ULP for bfloat16)
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat x) {
    x *= 2.0f;

    sfpi::vFloat j = x * sfpi::vConstFloatPrgm0;  // j = x * log2(e)
    sfpi::vFloat a = sfpi::setsgn(x, 0);
    // Rounds the absolute value of j, clamped to [0, 255].
    sfpi::vMag m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);
    sfpi::vInt i = m;

    sfpi::vFloat r, s, f, w, y, scale, bias, c0;

    if constexpr (!is_fp32_dest_acc_en) {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

        r = 8.361816406e-03f;
        r = r * f + 4.177856445e-02f;
        s = f * f;  // hide SFPMAD latency
        r = r * f + sfpi::vConstFloatPrgm2;
        c0 = 0.5f;
        r = __builtin_rvtt_sfpmad(r.get(), f.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    } else {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)_hi
        f = j * -1.42860677e-6f + f;         // f = f - j * ln(2)_lo

        r = 1.974105835e-04f;
        r = r * f + 1.393107930e-3f;
        r = r * f + 8.333439939e-3f;
        r = r * f + 4.166680202e-2f;
        s = f * f;  // hide SFPMAD latency
        r = r * f + sfpi::vConstFloatPrgm2;
        r = r * f + 4.999999702e-1f;
    }

    scale = sfpi::reinterpret<sfpi::vFloat>((i << 23) + sfpi::reinterpret<sfpi::vInt>(sfpi::vConst1));
    bias = scale - sfpi::vConst1;
    r = r * s + f;

    y = 1.0f;
    v_if(i < 61) {
        y = r * scale + bias;
        y = y * _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(y + 2.0f);
    }
    v_endif;

    y = sfpi::copysgn(y, x);

    return y;
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
                // Use accurate sigmoid-based tanh for fp32
                result = _sfpu_tanh_fp32_accurate_<is_fp32_dest_acc_en>(val);
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
            sinh_init<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
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
