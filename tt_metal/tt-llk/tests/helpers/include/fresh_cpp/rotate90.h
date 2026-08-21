// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `rotate90-fresh` coverage row (metal
// ckernel_sfpu_alt_complex_rotate90.h calculate_alt_complex_rotate90, corpus
// manifest class D-ABSENT — never dispatched by any test before this lane).
// Mathematical definition: the Dst tile holds interleaved complex numbers as
// (row 2k = real, row 2k+1 = imag) vector-row pairs; multiplying by i rotates
// each z = re + i*im by 90 degrees: i*z = -im + i*re, i.e.
//   out[2k]   = -in[2k+1]
//   out[2k+1] =  in[2k]
// Pure sign-flip + move at the value level — no arithmetic rounding, so the
// body is bit-preserving apart from the sign bit.
#include <cstdint>

namespace ckernel::sfpu
{

// ROW_PAIRS = vector-row pairs to rotate (16 = a full 32x32 bf16 tile).
template <int ROW_PAIRS>
__attribute__((noinline)) void calculate_rotate90_fresh_cpp()
{
    for (int d = 0; d < ROW_PAIRS; ++d)
    {
        const sfpi::vFloat re = sfpi::dst_reg[0];
        const sfpi::vFloat im = sfpi::dst_reg[1];
        sfpi::dst_reg[0]      = -im;
        sfpi::dst_reg[1]      = re;
        sfpi::dst_reg += 2;
    }
}

} // namespace ckernel::sfpu
