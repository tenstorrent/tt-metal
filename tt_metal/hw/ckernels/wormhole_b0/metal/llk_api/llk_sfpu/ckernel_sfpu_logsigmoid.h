// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"

namespace ckernel {
namespace sfpu {

// logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)).
//
// The exponential argument -|x| is never positive, so the exponential stays in (0, 1], cannot
// overflow, and feeds calculate_log1p_fp32 on its accurate range. This replaces the previous
// +/-4 piecewise split, which returned the raw input for x <= -4 (dropping the log1p(exp(x))
// residual), truncated the positive tail to -exp(-x), and carried a mid-range polynomial that
// is 7.12e-4 off at x = 0.
//
// dst_index_in1 is accepted but ignored: the exponential is computed internally from the input.
// The parameter is kept because SFPU_BINARY_CALL structurally passes three DST indices.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_logsigmoid(
    const std::uint32_t dst_index_in0,  // Index for input (x)
    const std::uint32_t dst_index_in1,  // Unused (kept for SFPU_BINARY_CALL arity)
    const std::uint32_t dst_index_out)  // Index for output
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

        // Seed with x so NaN payloads propagate through the final subtraction unchanged (the
        // min() predicate below is false for NaN); for non-NaN inputs this is min(x, 0).
        sfpi::vFloat result = x;
        v_if(x >= 0.0f) { result = 0.0f; }
        v_endif;

        sfpi::vFloat exp_neg_abs_x = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(-sfpi::abs(x));
        result = result - calculate_log1p_fp32<is_fp32_dest_acc_en>(exp_neg_abs_x);

        // Round-to-nearest into bf16 explicitly; a bare store would truncate the fp32 result.
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// calculate_log1p_fp32 reads the programmable SFPU constants configured by log1p_init, so those
// registers must be set up before calculate_logsigmoid runs. Delegating to log1p_init keeps the
// register values in sync with the log1p op by construction.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
void logsigmoid_init() {
    log1p_init<APPROXIMATION_MODE, false, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
