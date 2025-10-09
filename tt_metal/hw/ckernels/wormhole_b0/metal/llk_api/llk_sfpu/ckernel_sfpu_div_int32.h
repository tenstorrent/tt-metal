// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;

        sfpi::vInt in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vInt in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // sfpi::vInt in1_lo = in1 & 0xFFFF;
        // sfpi::vInt in1_hi = in1 >> 16;
        sfpi::vFloat in0_float = sfpi::reinterpret<sfpi::vFloat>(in0);
        sfpi::vFloat in1_float = sfpi::reinterpret<sfpi::vFloat>(in1);
        sfpi::vFloat recip = _sfpu_reciprocal_<2>(in1_float);
        // sfpi::vInt result = sfpi::reinterpret<sfpi::vInt>(recip);

        sfpi::vFloat result = in0_float * sfpi::setsgn(recip, in1);

        // sfpi::vFloat hi = sfpi::reinterpret<sfpi::vFloat>(in1);
        // sfpi::vFloat lo = sfpi::reinterpret<sfpi::vFloat>(in1 - sfpi::reinterpret<sfpi::vInt>(hi));

        // sfpi::vFloat result = _sfpu_reciprocal_<2>(hi);
        // sfpi::vInt result_int = sfpi::reinterpret<sfpi::vInt>(result);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;

        // // size of each tile in Dest is 64 rows
        // constexpr uint dst_tile_size = 64;
        // // operand A - int32
        // TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // // operand B - int32
        // TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);

        // // Extract sign bits of A and B
        // TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        // TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
        // TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    }
}

}  // namespace ckernel::sfpu
