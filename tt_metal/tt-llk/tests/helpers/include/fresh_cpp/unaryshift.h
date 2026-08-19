// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the unaryshift op (storm contract:
// fresh_cpp/README.md).  Independent derivation from the mathematical
// definition (the production golden: unbounded integer left shift by a
// fixed amount, exact int32 result; the kernel contract zeroes when the
// amount is outside [0, 32)):
//
//   left_shift(x) = (amt < 32) ? (x << amt) : 0
//
// on the two's-complement 32-bit view (typed DataLayout::I32 load/store,
// the production kernel's own contract).  The production hand-ism removed:
// the "#pragma GCC unroll 8" pin — a free loop.  The out-of-range zeroing
// is a host-scalar decision (amt is a dispatch constant), stated as a host
// branch, not a lane predicate.

#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_unary_shift_fresh_cpp(const std::uint32_t shift_amt)
{
    const unsigned amt = static_cast<unsigned>(shift_amt);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::vInt r       = 0;
        if (amt < 32u)
        {
            r = v << amt;
        }
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
