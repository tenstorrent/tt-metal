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
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr std::uint32_t dst_tile_size = 64;

        TT_SFPLOAD(0, INSTRUCTION_MODE, 3, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(1, INSTRUCTION_MODE, 3, dst_index_in1 * dst_tile_size);

        if constexpr (BITWISE_OP == BinaryBitwiseOp::AND) {
            TTI_SFPAND(0, 1, 0, 0);
        } else if constexpr (BITWISE_OP == BinaryBitwiseOp::OR) {
            TTI_SFPOR(0, 1, 0, 0);
        } else if constexpr (BITWISE_OP == BinaryBitwiseOp::XOR) {
            TTI_SFPXOR(0, 1, 0, 0);
        }

        TT_SFPSTORE(0, INSTRUCTION_MODE, 3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
