// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

enum class BinaryBitwiseOp : uint8_t
{
    AND = 0,
    OR  = 1,
    XOR = 2,
};

template <bool APPROXIMATION_MODE, BinaryBitwiseOp BITWISE_OP, InstrModLoadStore INSTRUCTION_MODE = INT32, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_bitwise_(const uint dst_offset)
{
    constexpr auto instruction_mode = static_cast<std::underlying_type_t<InstrModLoadStore>>(INSTRUCTION_MODE);
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr uint dst_tile_size = 64;

        TTI_SFPLOAD(0, instruction_mode, ADDR_MOD_7, 0);
        TT_SFPLOAD(1, instruction_mode, ADDR_MOD_7, dst_offset * dst_tile_size);

        if constexpr (BITWISE_OP == BinaryBitwiseOp::AND)
        {
            TTI_SFPAND(0, 1, 0, 0);
        }
        else if constexpr (BITWISE_OP == BinaryBitwiseOp::OR)
        {
            TTI_SFPOR(0, 1, 0, 0);
        }
        else if constexpr (BITWISE_OP == BinaryBitwiseOp::XOR)
        {
            TTI_SFPXOR(0, 1, 0, 0);
        }

        TTI_SFPSTORE(0, instruction_mode, ADDR_MOD_7, 0);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
