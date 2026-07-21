// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// SnakeBeta activation: y = x + sin²(α·x) / β.
// Range-reduces α·x to (-π/2, π/2] then evaluates sin(a) via the calculate_sine() minimax
// polynomial; sin² is even, so no quadrant sign-fix is needed. Valid for |α·x| < 32767·π
// (≈1.03e5) before convert<vSMag16> saturates.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void calculate_snake_beta(uint dst_index_x, uint dst_index_alpha, uint dst_index_beta, uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "snake_beta supports only Float32 and Float16_b");

    constexpr uint dst_tile_size_sfpi = 32;
    constexpr float one_over_pi = 0.318309886183791f;
    constexpr float pi_f = 3.141592653589793f;

    // sin(a) minimax coefficients on a ∈ (-π/2, π/2]; mirror calculate_sine().
    constexpr float fp32_C3 = 0x1.5dc908p-19f;
    constexpr float fp32_C2 = -0x1.9f70fp-13f;
    constexpr float fp32_C1 = 0x1.110edap-7f;
    constexpr float fp32_C0 = -0x1.55554cp-3f;
    constexpr float bf16_C2 = -0x1.8b10a4p-13f;
    constexpr float bf16_C1 = 0x1.10c2a2p-7f;
    constexpr float bf16_C0 = -0x1.5554a4p-3f;

    // 2 NR iters for fp32 (≤1 ULP), 1 for bf16 (≤0.5 ULP).
    constexpr int RECIP_ITER = is_fp32_dest_acc_en ? 2 : 1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[dst_index_x * dst_tile_size_sfpi];
        sfpi::vFloat alpha = sfpi::dst_reg[dst_index_alpha * dst_tile_size_sfpi];
        sfpi::vFloat beta = sfpi::dst_reg[dst_index_beta * dst_tile_size_sfpi];

        sfpi::vFloat ax = alpha * x;

        // a = (ax/π - round(ax/π)) * π, single-stage reduction; vConstFloatPrgm0/1/2 are
        // reserved for the reciprocal estimate so convert<vSMag16> is used instead.
        sfpi::vFloat ax_over_pi = ax * one_over_pi;
        sfpi::vSMag16 k = sfpi::convert<sfpi::vSMag16>(ax_over_pi, sfpi::RoundMode::Nearest);
        sfpi::vFloat k_f = sfpi::convert<sfpi::vFloat>(k, sfpi::RoundMode::Nearest);
        sfpi::vFloat a = (ax_over_pi - k_f) * pi_f;

        // sin(a) = a + a·s·poly(s), s = a².  PolynomialEvaluator::eval expands to a Horner chain.
        sfpi::vFloat s = a * a;
        sfpi::vFloat c = a * s;  // a³
        sfpi::vFloat r;
        if constexpr (is_fp32_dest_acc_en) {
            r = c * PolynomialEvaluator::eval(s, fp32_C0, fp32_C1, fp32_C2, fp32_C3) + a;
        } else {
            r = c * PolynomialEvaluator::eval(s, bf16_C0, bf16_C1, bf16_C2) + a;
        }

        sfpi::vFloat sin2_ax = r * r;
        sfpi::vFloat inv_beta = sfpu_reciprocal_iter<RECIP_ITER>(beta);
        sfpi::vFloat result = x + sin2_ax * inv_beta;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATE>
inline void snake_beta_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + counter reset), then the op-specific
    // reciprocal setup below -- one self-contained init, matching exp_init. snake_beta uses only
    // ADDR_MOD_7 (no op-specific ADDR_MOD_6).
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATE>();
}

}  // namespace ckernel::sfpu
