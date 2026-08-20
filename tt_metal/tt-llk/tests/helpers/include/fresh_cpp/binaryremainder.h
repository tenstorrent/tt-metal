// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `binaryremainder` corpus row (metal
// calculate_sfpu_binary_remainder, float arm).  Mathematical definition
// (torch.remainder): remainder(a, b) = a - floor(a/b) * b — the residue takes
// the DIVISOR's sign, |result| < |b|, remainder(a, 0) is NaN.  On magnitudes
// m = |a| mod |b| in [0, |b|): the result is m when the operand signs agree,
// |b| - m when they differ (and m != 0), with the divisor's sign folded on.
// The header's int/uint arms keep production-only coverage.
#include <cstdint>
#include <limits>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_binary_remainder_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat a  = sfpi::dst_reg[0];
            const sfpi::vFloat b  = sfpi::dst_reg[tile_rows];
            const sfpi::vFloat aa = sfpi::abs(a);
            const sfpi::vFloat ab = sfpi::abs(b);

            sfpi::vFloat r = fresh_mod_positive(aa, ab, fresh_recip_hwseed(ab));
            v_if (r - ab == 0.0f)
            {
                r = 0.0f;
            }
            v_endif;
            // Opposite operand signs fold the non-zero residue to the other
            // end of the divisor interval (floor vs trunc quotient).
            const sfpi::vFloat sign_product = sfpi::copysgn(sfpi::vFloat(1.0f), a) * sfpi::copysgn(sfpi::vFloat(1.0f), b);
            v_if (sign_product < 0.0f && r != 0.0f)
            {
                r = ab - r;
            }
            v_endif;
            // The result carries the divisor's sign; remainder(a, 0) is NaN.
            r = sfpi::copysgn(r, b);
            v_if (ab == 0.0f)
            {
                r = std::numeric_limits<float>::quiet_NaN();
            }
            v_endif;
            if constexpr (!DST_ACCUM_MODE)
            {
                r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = r;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_binary_remainder_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh binary remainder expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh binary remainder expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh binary remainder expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_binary_remainder_fresh_cpp<DST_ACCUM, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
