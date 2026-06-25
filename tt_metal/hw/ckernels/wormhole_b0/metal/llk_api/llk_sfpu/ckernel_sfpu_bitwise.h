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

enum class UnaryBitwiseOp : std::uint8_t {
    AND = 0,
    OR = 1,
    XOR = 2,
};

template <
    bool APPROXIMATION_MODE,
    UnaryBitwiseOp BITWISE_OP,
    InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32,
    int ITERATIONS = 8>
inline void calculate_sfpu_unary_bitwise(const uint value) {
    sfpi::vConstIntPrgm0 = value;  // LREG12 - scalar operand
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(0, INSTRUCTION_MODE, ADDR_MOD_3, 0);
        if constexpr (BITWISE_OP == UnaryBitwiseOp::AND) {
            TTI_SFPAND(0, p_sfpu::LREG12, 0, 0);
        } else if constexpr (BITWISE_OP == UnaryBitwiseOp::OR) {
            TTI_SFPOR(0, p_sfpu::LREG12, 0, 0);
        } else if constexpr (BITWISE_OP == UnaryBitwiseOp::XOR) {
            TTI_SFPXOR(0, p_sfpu::LREG12, 0, 0);
        }
        TTI_SFPSTORE(0, INSTRUCTION_MODE, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
