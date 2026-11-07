// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "sfpi.h"

// geglu_tanh(gate, up) = gelu_tanh(gate) * up in one SFPU pass; contract in api/compute/geglu_tanh.h.

namespace ckernel::sfpu {

// One Newton step on SFPARECIP's estimate: the denominator is a positive normal value. The 2.0f is a
// literal rather than a programmable constant, so it does not depend on what the init loads.
sfpi_inline sfpi::vFloat _geglu_tanh_reciprocal_(const sfpi::vFloat a) {
    sfpi::vFloat y = sfpi::approx_recip(a);
    sfpi::vFloat t = a * y - 2.0f;
    return y * -t - 0.0f;
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_geglu_tanh(
    const std::uint32_t gate_tile_idx, const std::uint32_t up_tile_idx, const std::uint32_t out_tile_idx) {
    using namespace sfpi;
    constexpr std::uint32_t dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat gate_x = dst_reg[gate_tile_idx * dst_tile_size];
        vFloat gate_a = sfpi::setsgn(gate_x, 0 /*sgn*/);
        vFloat gate_c2 = gate_x * gate_x;
        vFloat gate_g = gate_c2 * -0.10294324159622192f + -2.302208185195923f;
        // e^-|z| via the exponent ALU: w_c is its biased exponent (floored at 0.5 so it underflows to 0),
        // and the cubic in the fraction w_f is 2^fraction.
        vFloat w_c = gate_a * gate_g + 127.0f;
        vFloat w_z = sfpi::max(w_c, 0.5f);
        vInt w_m = sfpi::exman(w_z, sfpi::MantissaMode::ImplicitOne);
        vInt w_s = sfpi::shft(w_m, sfpi::exexp(w_z), sfpi::ShiftMode::Logical);
        vFloat w_y = sfpi::as<vFloat>(w_s);
        vMag w_q = sfpi::exman(w_y);
        vFloat w_f = sfpi::convert<vFloat>(w_q, RoundMode::Nearest);
        vFloat w = w_f * vConstFloatPrgm0 + 3.235038207097512e-15f;
        w = w * w_f + 8.286510677635306e-08f;
        w = w * w_f + 1.0f;
        w = sfpi::setexp(w, sfpi::exexp(w_y, ExponentMode::Biased));
        vFloat gate_n = gate_x;
        v_if(gate_x < 0.0f) { gate_n = gate_n * w; }
        v_endif;
        vFloat w1 = w + 1.0f;
        vFloat num = gate_n;
        vFloat up_x = dst_reg[up_tile_idx * dst_tile_size];
        // Every input's NaN reaches the product through a multiply-add.
        vFloat nan_z = 0.0f;
        vFloat den = w1;
        vFloat out = num * _geglu_tanh_reciprocal_(den);
        out = out * up_x + nan_z;
        if constexpr (!is_fp32_dest_acc_en) {
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        dst_reg[out_tile_idx * dst_tile_size] = out;
        dst_reg++;
    }
}

// The init resets the SFPU address counters and loads vConstFloatPrgm0 = the top coefficient of the 2^f polynomial. The
// body reads them, so geglu_tanh_init runs after any other SFPU op's init and before this kernel.
inline void geglu_tanh_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 1.305414094153441e-22f;
}

}  // namespace ckernel::sfpu
