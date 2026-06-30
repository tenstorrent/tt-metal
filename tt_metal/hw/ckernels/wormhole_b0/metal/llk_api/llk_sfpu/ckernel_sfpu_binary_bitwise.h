// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (INSTRUCTION_MODE == InstrModLoadStore::LO16) {
            vUInt a = dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U16>();
            vUInt b = dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::U16>();
            vUInt res;
            if constexpr (BITWISE_OP == BinaryBitwiseOp::AND) {
                res = a & b;
            } else if constexpr (BITWISE_OP == BinaryBitwiseOp::OR) {
                res = a | b;
            } else {
                res = a ^ b;
            }
            dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::U16>() = res;
        } else {
            vInt a = dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
            vInt b = dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
            vInt res;
            if constexpr (BITWISE_OP == BinaryBitwiseOp::AND) {
                res = a & b;
            } else if constexpr (BITWISE_OP == BinaryBitwiseOp::OR) {
                res = a | b;
            } else {
                res = a ^ b;
            }
            dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>() = res;
        }
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
