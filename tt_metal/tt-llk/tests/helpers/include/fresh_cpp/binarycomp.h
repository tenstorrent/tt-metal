// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `binarycomp` corpus row (metal
// ckernel_sfpu_binary_comp).  The row's nodes drive the SfpuElwEq arm:
// out = 1.0 where a == b, else 0.0 (the golden is float(t1 == t2)).  The
// Ne/int arms share the dispatch and keep production-only coverage.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_binary_comp_eq_fresh_cpp()
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
            sfpi::vFloat r       = 0.0f;
            // a == b stated as a - b == 0: equality of finite values is
            // exactly the zero difference (and +0 == -0 holds).
            v_if (a - b == 0.0f)
            {
                r = 1.0f;
            }
            v_endif;
            sfpi::dst_reg[0] = r;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_binary_comp_eq_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh binary eq expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh binary eq expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh binary eq expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_binary_comp_eq_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
