// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, InstrModLoadStore INSTRUCTION_MODE, int ITERATIONS>
inline void calculate_rsub_int(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Reverse subtract in 2's complement: out = in1 - in0 (B - A). `b - a` lowers to the same single
    // SFPIADD (2's-complement of A) the raw path used, but leaves the loads/store and dst walk to the
    // compiler. Pick the DataLayout whose load/store format byte matches the original
    // InstrModLoadStore (LO16->U16, INT32/2S_COMP->I32; on Blackhole INT32_2S_COMP is raw int32).
    constexpr sfpi::DataLayout layout =
        (INSTRUCTION_MODE == InstrModLoadStore::LO16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::I32;
    using vType = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
        vType b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() = b - a;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_rsub_scalar_int32(std::uint32_t scalar) {
    // out = scalar - dst. The scalar is materialized into a vInt once before the loop (as the raw
    // _sfpu_load_imm32_ path also did), and `s - a` lowers to the same single SFPIADD (2's-complement
    // of dst) per iteration, leaving the load/store and dst walk to the compiler.
    const sfpi::vInt s = static_cast<int>(scalar);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = s - a;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
