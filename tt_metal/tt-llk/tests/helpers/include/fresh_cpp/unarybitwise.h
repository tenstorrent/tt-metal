// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `unarybitwise-fresh` coverage row
// (metal ckernel_sfpu_bitwise.h calculate_sfpu_unary_bitwise, corpus manifest
// class D-ABSENT — the tested bitwise surface was only ever the BINARY
// header; this unary-with-scalar family had zero dispatch).  Mathematical
// definition (torch.bitwise_{and,or,xor} against an int scalar, Int32 view):
//   AND: y = x & c    OR: y = x | c    XOR: y = x ^ c
// exact bit-level contract, no rounding.
#include <cstdint>

namespace ckernel::sfpu
{

// SUBOP: 0 = AND, 1 = OR, 2 = XOR (the coverage vehicle races XOR; the three
// share one single-instruction mechanism class).
template <int SUBOP, int ITERATIONS>
__attribute__((noinline)) void calculate_unary_bitwise_fresh_cpp(const std::uint32_t value)
{
    const sfpi::vInt scalar = static_cast<int>(value);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vInt v = sfpi::dst_reg[0];
        sfpi::vInt r;
        if constexpr (SUBOP == 0)
        {
            r = v & scalar;
        }
        else if constexpr (SUBOP == 1)
        {
            r = v | scalar;
        }
        else
        {
            static_assert(SUBOP == 2, "unary bitwise SUBOP is AND(0)/OR(1)/XOR(2)");
            r = v ^ scalar;
        }
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
