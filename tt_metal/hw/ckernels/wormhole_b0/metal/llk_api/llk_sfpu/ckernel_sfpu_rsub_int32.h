// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include <type_traits>
#include "sfpu/ckernel_sfpu_load_config.h"
namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void calculate_rsub_int(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");
    // Each Dest tile is 64 rows; sfpi dst_reg[] indexes in stride units (SFP_DESTREG_STRIDE == 2),
    // so 64 raw rows == 32 sfpi stride units.
    constexpr uint dst_tile_size = 32;

    // Reverse subtract: out = in1 - in0. sfpi's `b - a` lowers to the same SFPIADD that 2's-complements
    // the subtrahend, matching the original TTI_SFPIADD(..., 2SCOMP_LREG_DST). The load/store DataLayout
    // is chosen so its SFP load/store format byte equals the original InstrModLoadStore value:
    //   INT32 (4) -> I32 (sign-mag<->2's-comp conversion), LO16 (6) -> U16, INT32_2S_COMP (12) -> SM32 (raw).
    constexpr sfpi::DataLayout layout = (INSTRUCTION_MODE == InstrModLoadStore::LO16)            ? sfpi::DataLayout::U16
                                        : (INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP) ? sfpi::DataLayout::SM32
                                                                                                 : sfpi::DataLayout::I32;
    using vType = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size].mode<layout>();
        vType b = sfpi::dst_reg[dst_index_in1 * dst_tile_size].mode<layout>();
        sfpi::dst_reg[dst_index_out * dst_tile_size].mode<layout>() = b - a;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_rsub_scalar_int32(uint32_t scalar) {
    // out = scalar - x. The immediate is materialized once (loop-invariant) outside the loop.
    sfpi::vInt scalar_vec = static_cast<int>(scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt x = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = scalar_vec - x;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
