// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log1p.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_logaddexp_(sfpi::vFloat a, sfpi::vFloat b) {
    // Numerically stable logaddexp: max(a,b) + log1p(exp(-|a-b|))
    auto [min_val, max_val] = sfpi::min_max(a, b);
    sfpi::vFloat diff = max_val - min_val;

    // Compute exp(-diff). Since diff >= 0, this is always in (0, 1].
    sfpi::vFloat neg_diff = -diff;
    sfpi::vFloat exp_neg_diff = sfpi::exp(neg_diff);

    // log1p(exp(-diff)) using the existing log1p implementation
    sfpi::vFloat log1p_result = calculate_log1p_fp32<is_fp32_dest_acc_en>(exp_neg_diff);

    return max_val + log1p_result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_logaddexp2_(sfpi::vFloat a, sfpi::vFloat b) {
    // log2(2^a + 2^b) = logaddexp(a*ln2, b*ln2) / ln2
    // Stable form: max(a,b) + log2(1 + 2^(-|a-b|))
    //            = max(a,b) + log1p(2^(-|a-b|)) / ln2
    auto [min_val, max_val] = sfpi::min_max(a, b);
    sfpi::vFloat diff = max_val - min_val;

    // 2^(-diff) = exp(-diff * ln2)
    constexpr float LN2 = 0.6931471805599453f;
    sfpi::vFloat neg_diff_ln2 = -diff * LN2;
    sfpi::vFloat exp_neg_diff_ln2 = sfpi::exp(neg_diff_ln2);

    sfpi::vFloat log1p_result = calculate_log1p_fp32<is_fp32_dest_acc_en>(exp_neg_diff_ln2);

    // Divide by ln2 to convert from natural log to log2
    constexpr float INV_LN2 = 1.4426950408889634f;
    return max_val + log1p_result * INV_LN2;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = _sfpu_logaddexp_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in0, in1);
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp2(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = _sfpu_logaddexp2_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in0, in1);
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp_init() {
    log1p_init<APPROXIMATION_MODE, false, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_logaddexp2_init() {
    log1p_init<APPROXIMATION_MODE, false, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
