// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "sfpi.h"

// quick_gelu(x) = x / (1 + exp(-1.702 x)) for BF16 inputs on Blackhole: at most 0.976 ULP over every finite BF16
// input in either DEST mode (FP32 DEST results are within 2.1e-03 relative). A NaN input stores +inf with BF16 DEST.

namespace ckernel::sfpu {

// One Newton step on SFPARECIP's estimate: the denominator is a positive normal value. The 2.0f is a
// literal rather than a programmable constant, so it does not depend on what the init loads.
sfpi_inline sfpi::vFloat _quick_gelu_reciprocal_(const sfpi::vFloat a) {
    sfpi::vFloat y = sfpi::approx_recip(a);
    sfpi::vFloat t = a * y - 2.0f;
    return y * -t - 0.0f;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_quick_gelu() {
    using namespace sfpi;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat input_x = dst_reg[0];
        vFloat input_a = sfpi::setsgn(input_x, 0 /*sgn*/);
        // e^-|z| through the exponent ALU: w_c is the biased exponent of e^-|z|, w_f its fraction, w = 2^fraction.
        vFloat w_c = input_a * vConstFloatPrgm0 + 127.0f;
        vFloat w_z = sfpi::max(w_c, 0.5f);
        vInt w_m = sfpi::exman(w_z, sfpi::MantissaMode::ImplicitOne);
        vInt w_s = sfpi::shft(w_m, sfpi::exexp(w_z), sfpi::ShiftMode::Logical);
        vFloat w_y = sfpi::as<vFloat>(w_s);
        vMag w_q = sfpi::exman(w_y);
        vFloat w_f = sfpi::convert<vFloat>(w_q, RoundMode::Nearest);
        vFloat w = w_f * vConstFloatPrgm1 + 7.939760848785227e-08f;
        w = w * w_f + 1.0f;
        w = sfpi::setexp(w, sfpi::exexp(w_y, ExponentMode::Biased));
        vFloat input_n = input_x;
        v_if(input_x < 0.0f) { input_n = input_n * w; }
        v_endif;
        vFloat w1 = w + 1.0f;
        vFloat num = input_n;
        // Every input's NaN reaches the product through a multiply-add.
        vFloat nan_z = 0.0f;
        vFloat den = w1;
        vFloat out = num * _quick_gelu_reciprocal_(den) + nan_z;
        if constexpr (!is_fp32_dest_acc_en) {
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        dst_reg[0] = out;
        dst_reg++;
    }
}

// Prgm0 = -1.702 * log2(e), Prgm1 = top coefficient of the 2^f polynomial; run after any other SFPU op's init.
inline void quick_gelu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = -2.4554669857025146f;
    sfpi::vConstFloatPrgm1 = 4.686561299357527e-15f;
}

}  // namespace ckernel::sfpu
