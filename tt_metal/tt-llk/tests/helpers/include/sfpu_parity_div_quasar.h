// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu/ckernel_sfpu_div_int32.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"

namespace ckernel::sfpu
{

// Float32 transport keeps the existing fuser format contract. Construct numeric integers,
// invoke the real integer-input/float-output division, and restore both transport inputs.
// numerator = round_even(16*a)-8; denominator = sign(b)*(round_even(abs(16*b))+1).
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_parity_div_int32_float()
{
    constexpr uint stride = (1U << trisc::get_dest_tile_size_log2(trisc::DstTileShape::Tile32x32)) / sfpi::SFP_DESTREG_STRIDE;
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat original_a = sfpi::dst_reg[0];
        sfpi::vFloat original_b = sfpi::dst_reg[stride];
        // Preserve FP32 tails near half-integers; SFPCAST can round these as exact ties.
        // The native FP32 RNE helper is valid for |16*input| < 2^22.
        sfpi::vInt numerator;
        sfpi::vInt denominator;
        _sfpu_round_to_nearest_int32_(original_a * 16.0f, numerator);
        _sfpu_round_to_nearest_int32_(sfpi::abs(original_b * 16.0f), denominator);
        numerator -= 8;
        denominator += 1;
        v_if (original_b < 0.0f)
        {
            denominator = -denominator;
        }
        v_endif;
        sfpi::dst_reg[0].mode<sfpi::DataLayout::SM32>()      = numerator;
        sfpi::dst_reg[stride].mode<sfpi::DataLayout::SM32>() = denominator;
        calculate_div_int32_float_body<APPROXIMATION_MODE>(0, 1, 2);
        sfpi::dst_reg[0]      = original_a;
        sfpi::dst_reg[stride] = original_b;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
