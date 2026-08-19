// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the square op (storm contract:
// fresh_cpp/README.md).  Independent derivation from the mathematical
// definition (the production golden is plain x*x with IEEE overflow to inf):
//
//   square(x) = x * x
//
// The production kernel pins "#pragma GCC unroll 0" on the loop — the
// hand-ism this body removes: a free loop whose unrolling, pipelining and
// Dst delivery are the compiler's.

#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_square_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0]     = x * x;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
