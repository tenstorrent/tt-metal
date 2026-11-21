// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

enum class BinaryBitwiseOp : std::uint8_t {
    AND = 0,
    OR = 1,
    XOR = 2,
};

// Programs ADDR_MOD_6 with dest.incr = SFP_ROWS so the store in calculate_sfpu_binary_bitwise advances
// the destination by one SFPU pass (what dst_reg++ did), in place of a separate TTINCRWC per pass.
// Quasar's shared SFPU init only programs ADDR_MOD_7 (incr = 0); this is additive. Called by
// binary_bitwise_tile_init (and the tt-llk harness init) for the three bitwise ops.
inline void binary_bitwise_init() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = ckernel::math::SFP_ROWS},
    }
        .set(ADDR_MOD_6);
}

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
        (INSTRUCTION_MODE == InstrModLoadStore::UINT16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::I32;
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

        // The store advances the destination by one SFPU pass through ADDR_MOD_6 (dest += SFP_ROWS, what
        // dst_reg++ does), programmed by binary_bitwise_init; it replaces the separate TTINCRWC per pass.
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<layout>(ADDR_MOD_6) = result;
    }
}

}  // namespace sfpu
}  // namespace ckernel
