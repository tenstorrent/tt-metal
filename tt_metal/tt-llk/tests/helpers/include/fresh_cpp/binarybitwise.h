// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `binarybitwise` corpus row (metal
// calculate_sfpu_binary_bitwise).  The row's nodes drive the AND arm:
// out = a & b on the raw int32 bit patterns (the golden is
// torch.bitwise_and over int32; the production path reads Dst raw, so the
// typed raw-I32 Dst view carries the same representation contract).  The
// OR/XOR arms share the dispatch and keep production-only coverage.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_binary_bitwise_and_fresh_cpp()
{
    constexpr std::uint32_t tile_rows = 32;

#pragma GCC unroll 4
    for (int face = 0; face < 4; ++face)
    {
#pragma GCC unroll 8
        for (int row = 0; row < ITERATIONS; ++row)
        {
            const sfpi::vInt a                             = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            const sfpi::vInt b                             = sfpi::dst_reg[tile_rows].mode<sfpi::DataLayout::I32>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = a & b;
            sfpi::dst_reg++;
        }
        ::_llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS>
inline void call_binary_bitwise_and_fresh_cpp(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out, const VectorMode vector_mode)
{
    ::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    LLK_ASSERT(dst_index_in1 == dst_index_in0 + 1, "fresh bitwise and expects adjacent inputs");
    LLK_ASSERT(dst_index_out == dst_index_in0, "fresh bitwise and expects in-place output");
    LLK_ASSERT(vector_mode == VectorMode::RC, "fresh bitwise and expects full-tile vector mode");

    ::_llk_math_eltwise_sfpu_start_(dst_index_in0);
    calculate_binary_bitwise_and_fresh_cpp<ITERATIONS>();
    ::_llk_math_eltwise_sfpu_done_();
}

} // namespace ckernel::sfpu
