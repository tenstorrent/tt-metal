// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `copydest-fresh` coverage row (metal
// ckernel_sfpu_copy_dest_values.h copy_dest_value, corpus manifest class
// D-ABSENT — zero dispatch anywhere).  Mathematical definition: identity
// tile move between Dst-resident tiles, out[i] = in[i] — value-preserving
// (a bf16 load/store round-trip is exact), no arithmetic.  Full-tile body:
// 4 faces x ITERATIONS rows, output tile at the +32 vector-row Dst offset
// (source tile 0 -> destination tile 1, the inverse dataflow of the fresh
// binary idiom's in-place output).
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_copy_dest_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vFloat v     = sfpi::dst_reg[0];
            sfpi::dst_reg[tile_rows] = v;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_copy_dest_fresh_cpp(const std::uint32_t dst_index_in, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in, dst_index_out, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_out == dst_index_in + 1, "fresh copy-dest expects the adjacent output tile");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh copy-dest expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in);
    calculate_copy_dest_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
