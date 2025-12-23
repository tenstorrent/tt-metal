// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_div_int32_floor.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_remainder_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    // Use a temporary index to store the quotient from floor division
    // We'll use the output index as temporary since we'll overwrite it anyway
    // constexpr uint tmp_quotient_index = dst_index_out;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // First, compute floor division to get the quotient
        calculate_div_int32_body<true>(dst_index_in0, dst_index_in1, dst_index_out);

        // Load the quotient, in0, and in1
        // Equivalent to: sfpi::vInt quotient = sfpi::dst_reg[tmp_quotient_index * dst_tile_size_sfpi];
        sfpi::vInt quotient = __builtin_rvtt_sfpload(
            4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].get());

        // Reload in0 and in1
        sfpi::vInt in0 = __builtin_rvtt_sfpload(
            4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());
        sfpi::vInt in1 = __builtin_rvtt_sfpload(
            4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].get());

        // Compute remainder: remainder = in0 - quotient * in1
        sfpi::vInt remainder = in0 - quotient * in1;

        // Store the remainder result
        __builtin_rvtt_sfpstore(
            remainder.get(), 4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].get());

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void remainder_int32_init() {
    // Use the same initialization as div_floor_init since we're using the floor division kernel
    div_floor_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
