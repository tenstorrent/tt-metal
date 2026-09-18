// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"

#include "ckernel_trisc_common.h"

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
        INSTRUCTION_MODE == InstrModLoadStore::INT32_2S_COMP || INSTRUCTION_MODE == InstrModLoadStore::INT32 ||
            INSTRUCTION_MODE == InstrModLoadStore::LO16,
        "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Quasar Int32 Dest is sign-magnitude; SM32 converts to/from vInt arithmetic.
    // Preserve I32 only for callers explicitly requesting raw two's-complement storage.
    constexpr sfpi::DataLayout layout = (INSTRUCTION_MODE == InstrModLoadStore::LO16)    ? sfpi::DataLayout::U16
                                        : (INSTRUCTION_MODE == InstrModLoadStore::INT32) ? sfpi::DataLayout::SM32
                                                                                         : sfpi::DataLayout::I32;
    using vType = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

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
        sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>() = s - a;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
