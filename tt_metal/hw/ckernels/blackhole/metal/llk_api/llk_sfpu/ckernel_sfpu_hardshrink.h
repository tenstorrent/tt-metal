// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardshrink(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint32_t param0) {
    // Hardshrink(x, λ) = x if |x| > λ, else 0
    // Single comparison using abs: setsgn(v, 0) clears sign bit
    // param0 contains lambda as FP32 bits. For BF16 inputs, the host pre-rounds
    // lambda to BF16 precision (then re-expands to FP32) so that FP32→FP19b
    // truncation on SFPU preserves the BF16 value exactly. For FP32 inputs,
    // lambda is passed as full FP32, so both input and lambda undergo the same
    // FP32→FP19b truncation, keeping comparisons consistent.
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    sfpi::vFloat lambda = Converter::as_float(param0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];
        sfpi::vFloat abs_v = sfpi::setsgn(v, 0);
        v_if(abs_v <= lambda) { sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = sfpi::vConst0; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
