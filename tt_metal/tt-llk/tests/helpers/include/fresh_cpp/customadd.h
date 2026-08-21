// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `customadd-fresh` coverage row (metal
// experimental ckernel_sfpu_custom_add.h my_add_tile_face, corpus manifest
// class D-ABSENT — zero dispatch anywhere).  Mathematical definition:
// elementwise fp32 addition of two Dst-resident tiles, out = a + b, with the
// RNE cast of the golden's final bf16 store (binary-float fresh precedent).
// Full-tile body: 4 faces x ITERATIONS rows, operand tile 1 at the +32
// vector-row Dst offset (the fresh binary idiom).
#include <cstdint>

namespace ckernel::sfpu
{

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_custom_add_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat a = sfpi::dst_reg[0];
            const sfpi::vFloat b = sfpi::dst_reg[tile_rows];
            sfpi::vFloat r       = a + b;
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
inline void call_custom_add_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh custom add expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh custom add expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh custom add expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_custom_add_fresh_cpp<DST_ACCUM, ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
