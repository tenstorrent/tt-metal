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
        // Note: This kernel is a work in progress due to a known issue in BH #33044

        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vInt in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vInt in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat float_in0 = sfpi::int32_to_float(in0, 0);
        sfpi::vFloat float_in1 = sfpi::int32_to_float(in1, 0);
        sfpi::dst_reg[0] = float_in0;
        sfpi::dst_reg++;

        // constexpr uint dst_tile_size_sfpu = 64;

        // // Typecasting input A and B without using sfpi::int32_to_float
        // // Typecast input A to fp32
        // TT_SFPLOAD(0, INT32_2S_COMP, ADDR_MOD_7, dst_index_in0 * dst_tile_size_sfpu);
        // TT_SFPCAST(0, 2, INT_SIGN_MAGN_TO_INT32_2S_COMP);
        // // Required after cast due to a bug in Blackhole RTL.
        // TT_SFPSETSGN(0, 2, 0, 0);
        // TT_SFPCAST(0, 1, 0);
        // TT_SFPSTORE(1, 3, ADDR_MOD_7, 0);

        // // Store typecasted tensor to float_in0
        // sfpi::vFloat float_in0 = sfpi::dst_reg[0];

        // // Typecast input B to fp32
        // TT_SFPLOAD(0, INT32_2S_COMP, ADDR_MOD_7, dst_index_in1 * dst_tile_size_sfpu);
        // TT_SFPCAST(0, 2, INT_SIGN_MAGN_TO_INT32_2S_COMP);
        // // Required after cast due to a bug in Blackhole RTL.
        // TT_SFPSETSGN(0, 2, 0, 0);
        // TT_SFPCAST(0, 1, 0);
        // TT_SFPSTORE(1, 3, ADDR_MOD_7, 0);

        // // Store typecasted tensor to float_in1
        // sfpi::vFloat float_in1 = sfpi::dst_reg[0];

        // sfpi::vFloat result = _sfpu_reciprocal_<2>(float_in1);
        // // sfpi::vFloat result = float_in0 * sfpi::setsgn(_sfpu_reciprocal_<2>(float_in1), float_in1);
    }
}

template <bool APPROXIMATION_MODE>
inline void div_init() {
    _init_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
