// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

enum class UnaryBitwiseOp : std::uint8_t {
    AND = 0,
    OR = 1,
    XOR = 2,
};

template <UnaryBitwiseOp BITWISE_OP, typename Vec>
inline Vec compute_unary_bitwise(Vec v, Vec scalar) {
    if constexpr (BITWISE_OP == UnaryBitwiseOp::AND) {
        return v & scalar;
    } else if constexpr (BITWISE_OP == UnaryBitwiseOp::OR) {
        return v | scalar;
    } else {
        static_assert(BITWISE_OP == UnaryBitwiseOp::XOR, "Unsupported bitwise op");
        return v ^ scalar;
    }
}

template <
    bool APPROXIMATION_MODE,
    UnaryBitwiseOp BITWISE_OP,
    DataFormat DATA_FORMAT = DataFormat::Int32,
    int ITERATIONS = 8>
inline void calculate_sfpu_unary_bitwise(const uint value) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for bitwise operation. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::UInt16) {
        // Bitwise op is per-lane (v & scalar), so `scalar` must be a vector broadcast of `value`
        // whose type matches the loaded `v` (vUInt for the U16 layout).
        const sfpi::vUInt scalar = static_cast<unsigned>(value);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vUInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::U16>() = compute_unary_bitwise<BITWISE_OP>(v, scalar);
            sfpi::dst_reg++;
        }
    } else {
        // I32 layout loads vInt, so `scalar` must be vInt to match `v`.
        const sfpi::vInt scalar = static_cast<int>(value);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = compute_unary_bitwise<BITWISE_OP>(v, scalar);
            sfpi::dst_reg++;
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Bitwise<APPROX, BITWISE_OP, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode, value) -> calculate_sfpu_unary_bitwise
//                                               (bitwise_and_tile, bitwise_or_tile, bitwise_xor_tile)
//   init()                                   -> bare init (bitwise_{and,or,xor}_tile_init)
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    UnaryBitwiseOp BITWISE_OP,
    DataFormat FORMAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct Bitwise : SfpuUnaryOp<
                     Bitwise<APPROXIMATION_MODE, BITWISE_OP, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>,
                     DST_SYNC,
                     DST_ACCUM> {
    static void kernel(std::uint32_t value) {
        calculate_sfpu_unary_bitwise<APPROXIMATION_MODE, BITWISE_OP, FORMAT, ITERATIONS>(value);
    }
};
}  // namespace sfpu
}  // namespace ckernel
