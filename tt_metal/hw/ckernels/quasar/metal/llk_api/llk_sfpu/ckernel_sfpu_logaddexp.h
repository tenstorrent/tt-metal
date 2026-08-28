// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel::sfpu {

// logaddexp(a, b) = max(a, b) + log1p(exp(-|a - b|))
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = sfpi::max(a, b);

        // Guard against inf - inf = NaN.
        // In SFPU, inf == inf is True, NaN == NaN is False.
        // By replacing identical inputs with 0.0f difference, we correctly
        // output max(inf, inf) + ln(2) = inf, and NaN naturally propagates.
        v_if(a == b) {
            a = 0.0f;
        }
        v_else {
            a = sfpi::abs(a - b);
        }
        v_endif;

        // b is re-used to avoid spilling registers. Maximum 3 vFloat live at once.
        b = _sfpu_exp_fp32_accurate_(-a);
        result = result + calculate_log1p_fp32<is_fp32_dest_acc_en>(b);

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// logaddexp2(a, b) = max(a, b) + log2(1 + 2^(-|a - b|))
// Formulated as max(a, b) + log1p(exp(-|a - b| * ln(2))) / ln(2)
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_logaddexp2(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = sfpi::max(a, b);

        v_if(a == b) {
            a = 0.0f;
        }
        v_else {
            a = sfpi::abs(a - b);
        }
        v_endif;

        constexpr float LN2 = 0.6931471824645996f;
        constexpr float INV_LN2 = 1.4426950408889634f;

        b = _sfpu_exp_fp32_accurate_(-a * LN2);
        result = result + calculate_log1p_fp32<is_fp32_dest_acc_en>(b) * INV_LN2;

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Initialization parameters for log1p.
template <bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp_init() {
    const float LOG_TWO = 0.693147182f;
    const float TWO_TO_M23 = 1.19209290e-7f;
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
        sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
    } else {
        sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
        sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
    }
}

}  // namespace ckernel::sfpu
