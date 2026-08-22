// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_logsigmoid(
    const std::uint32_t dst_index_in0,  // Index for input (x)
    const std::uint32_t dst_index_in1,  // Index for exp(-x) (unused by this implementation)
    const std::uint32_t dst_index_out)  // Index for output
{
    // logsigmoid(x) = min(x, 0) - log1p(exp(-|x|))
    //
    // The exponential input is never positive, so the intermediate cannot
    // overflow.  Both primitives already exist: _sfpu_exp_fp32_accurate_ in
    // ckernel_sfpu_exp.h and calculate_log1p_fp32 in ckernel_sfpu_log1p.h.
    //
    // The pre-computed exp(-x) operand (dst_index_in1) is not used; this
    // implementation computes exp(-|x|) directly for numerical stability.
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    // Always use the fp32 polynomial path in calculate_log1p_fp32.  This is
    // correct for both fp32 and bf16 DEST (the fp32 path is a superset) and
    // avoids threading DST_ACCUM_MODE through the binary SFPU call macro.
    // Must match the third template argument of log1p_init in logsigmoid_init.
    constexpr bool is_fp32_dest_acc_en = true;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

        // exp(-|x|): input is always <= 0, so no overflow
        sfpi::vFloat neg_abs_x = -sfpi::abs(x);
        sfpi::vFloat exp_neg_abs_x = _sfpu_exp_fp32_accurate_(neg_abs_x);

        // log1p(exp(-|x|))
        sfpi::vFloat log1p_val = calculate_log1p_fp32<is_fp32_dest_acc_en>(exp_neg_abs_x);

        // min(x, 0) - log1p(exp(-|x|))
        sfpi::vFloat result = -log1p_val;
        v_if(x < 0.0f) { result = x - log1p_val; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void logsigmoid_init() {
    // log1p requires vConstFloatPrgm0/1/2 to be initialised
    log1p_init<APPROXIMATION_MODE, false, true>();
}

}  // namespace sfpu
}  // namespace ckernel
