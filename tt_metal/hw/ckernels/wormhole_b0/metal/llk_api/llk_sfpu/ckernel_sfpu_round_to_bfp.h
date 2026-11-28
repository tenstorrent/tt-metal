// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// One value of a block, rounded the way the host's convert_u32_to_bfp does it: the 24-bit significand is
// shifted down to the block exponent (dropping the bits shifted out), rounded to MANT_BITS bits to nearest
// with ties to even, and clamped to the largest mantissa. block_scale is 2^(block_exp - 127).
template <std::uint32_t MANT_BITS>
sfpi_inline sfpi::vFloat _round_to_bfp_value_(sfpi::vFloat x, sfpi::vInt block_exp, sfpi::vFloat block_scale) {
    constexpr std::uint32_t SHIFT = 24 - MANT_BITS;

    sfpi::vInt align = sfpi::exexp(x, sfpi::ExponentMode::Biased) - block_exp;
    sfpi::vUInt aligned = sfpi::shft(sfpi::as<sfpi::vUInt>(sfpi::exman(x, sfpi::MantissaMode::WithUnitBit)), align);
    // SFPSHFT only uses the low 5 bits of the amount. A zero has exponent 0, so it lands here too as long as
    // block_exp > 31.
    v_if(align < -31) { aligned = 0; }
    v_endif;

    sfpi::vUInt mant = (aligned + ((1u << (SHIFT - 1)) - 1) + ((aligned >> SHIFT) & 1)) >> SHIFT;
    // Rounding up can only reach 2^MANT_BITS; take that back to the largest mantissa.
    mant = mant - (mant >> MANT_BITS);

    // 2 + mant * 2^(1 - MANT_BITS) as a float; less 2, times 2^(block_exp - 127), is mant times the block's step.
    sfpi::vFloat result = sfpi::as<sfpi::vFloat>((mant << (23 - MANT_BITS)) | 0x40000000u);
    return sfpi::copysgn((result - 2.0f) * block_scale, x);
}

// Rounds every value of a face to the grid of the block-float format the packer then writes (MANT_BITS = 7 for
// Bfp8_b, 3 for Bfp4_b), so the pack has nothing left to round and gives the host's bits. A block is one Dest row
// of 16 values sharing the largest exponent among them. One SFPLOAD holds the even or odd columns of four rows,
// eight lanes per row, so the block maximum is the lanewise max of the even and odd loads followed by a butterfly
// over each group of eight lanes. Blocks of inf/NaN, or whose largest value is below 2^-95, are left to the packer.
template <bool APPROXIMATION_MODE, int ITERATIONS, std::uint32_t MANT_BITS>
inline void calculate_round_to_bfp() {
    static_assert(MANT_BITS == 7 || MANT_BITS == 3, "Bfp8_b keeps 7 mantissa bits, Bfp4_b 3");
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat even = sfpi::dst_reg[0];
        sfpi::vFloat odd = sfpi::dst_reg[1];

        sfpi::vFloat block_max = sfpi::max(sfpi::abs(even), sfpi::abs(odd));
        sfpi::vFloat rotated = sfpi::subvec_shflror1(block_max);
        block_max = sfpi::max(block_max, rotated);
        rotated = sfpi::subvec_shflror1(block_max);
        rotated = sfpi::subvec_shflror1(rotated);
        block_max = sfpi::max(block_max, rotated);
        rotated = sfpi::subvec_shflror1(block_max);
        rotated = sfpi::subvec_shflror1(rotated);
        rotated = sfpi::subvec_shflror1(rotated);
        rotated = sfpi::subvec_shflror1(rotated);
        block_max = sfpi::max(block_max, rotated);
        sfpi::vInt block_exp = sfpi::exexp(block_max, sfpi::ExponentMode::Biased);

        v_if(block_exp > 31 && block_exp < 255) {
            sfpi::vFloat block_scale = sfpi::setexp(sfpi::vFloat(1.0f), block_exp);
            sfpi::dst_reg[0] = _round_to_bfp_value_<MANT_BITS>(even, block_exp, block_scale);
            sfpi::dst_reg[1] = _round_to_bfp_value_<MANT_BITS>(odd, block_exp, block_scale);
        }
        v_endif;
        sfpi::dst_reg += 2;
    }
}

}  // namespace sfpu
}  // namespace ckernel
