// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "llk_defs.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

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
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    // Bitwise AND/OR/XOR of two integer operands. `a & b` / `a | b` / `a ^ b` lower to the same
    // single SFPAND/SFPOR/SFPXOR the raw path used (no constant materialization), while the loads,
    // store and dst walk are left to the compiler.
    constexpr sfpi::DataLayout layout =
        (INSTRUCTION_MODE == InstrModLoadStore::LO16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::I32;
    using vType = std::conditional_t<layout == sfpi::DataLayout::U16, sfpi::vUInt, sfpi::vInt>;

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vType a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<layout>();
        vType b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<layout>();

        vType result;
        if constexpr (BITWISE_OP == BinaryBitwiseOp::AND) {
            result = a & b;
        } else if constexpr (BITWISE_OP == BinaryBitwiseOp::OR) {
            result = a | b;
        } else {
            result = a ^ b;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>() = result;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// BinaryBitwise<APPROXIMATION_MODE, BITWISE_OP, INSTRUCTION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(in0, in1, out, vector_mode) -> calculate_sfpu_binary_bitwise; init() is the shared bare init.
//   Backs bitwise_{and,or,xor}_binary_tile / binary_bitwise_tile_init (api/compute/binary_bitwise_sfpu.h).
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    BinaryBitwiseOp BITWISE_OP,
    InstrModLoadStore INSTRUCTION_MODE,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct BinaryBitwise
    : SfpuBinaryOp<
          BinaryBitwise<APPROXIMATION_MODE, BITWISE_OP, INSTRUCTION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>,
          DST_SYNC,
          DST_ACCUM> {
    static void kernel(std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out) {
        calculate_sfpu_binary_bitwise<APPROXIMATION_MODE, BITWISE_OP, INSTRUCTION_MODE, ITERATIONS>(
            dst_index_in0, dst_index_in1, dst_index_out);
    }
};
}  // namespace sfpu
}  // namespace ckernel
