// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

enum class BinaryBitwiseOp : std::uint8_t {
    AND = 0,
    OR = 1,
    XOR = 2,
};

template <
    bool APPROXIMATION_MODE,
    BinaryBitwiseOp BITWISE_OP,
    InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32,
    int ITERATIONS = 8>
inline void calculate_sfpu_binary_bitwise(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(
        INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16,
        "INSTRUCTION_MODE must be one of: INT32, LO16.");
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    // LO16 uses unsigned 16-bit (U16) layout, INT32 uses two's-complement 32-bit (I32) layout.
    constexpr auto layout =
        (INSTRUCTION_MODE == InstrModLoadStore::LO16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::I32;
    using vType = std::conditional_t<INSTRUCTION_MODE == InstrModLoadStore::LO16, vUInt, vInt>;
    auto apply = [](vType a, vType b) -> vType {
        if constexpr (BITWISE_OP == BinaryBitwiseOp::AND) {
            return a & b;
        } else if constexpr (BITWISE_OP == BinaryBitwiseOp::OR) {
            return a | b;
        } else {
            return a ^ b;
        }
    };
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() =
            apply(dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>(),
                  dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>());
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
