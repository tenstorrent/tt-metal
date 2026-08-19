// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `binary-float` corpus row (metal
// ckernel_sfpu_binary float path).  The row's nodes drive the SfpuElwsub arm:
// out = a - b computed in fp32 (the golden widens to fp32 and casts back);
// for a 16-bit destination the result is rounded to nearest-even before the
// truncating store, which is the golden's final cast.  The header's other
// float mathops (add/mul/div/pow/rsub) share the dispatch and keep their
// production-only coverage — one representative mathop per row.
#include <cstdint>

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_binary_float_sub_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat lhs = sfpi::dst_reg[0];
            const sfpi::vFloat rhs = sfpi::dst_reg[tile_rows];
            sfpi::vFloat r         = lhs - rhs;
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
inline void call_binary_float_sub_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh binary sub expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh binary sub expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh binary sub expects full-tile vector mode");

    // Anchor the dynamic tile once in the wrapper so the isolated semantic
    // body contains only constant relative Dst addresses (fresh max/min idiom).
    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_binary_float_sub_fresh_cpp<DST_ACCUM, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
