// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_sfpu_o_norm.h"
#include "llk_math_eltwise_ternary_sfpu.h"

namespace ckernel {

template <bool APPROXIMATION_MODE>
inline void llk_math_o_norm_sfpu_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::o_norm>();
    sfpu::o_norm_init<APPROXIMATION_MODE>();
}

// Fused o_norm over a group of NUM_REDUCE_TILES tiles per operand. Reads o at
// dst_index_in0, gamma2 at dst_index_in1, g_out at dst_index_in2, and writes
// the result to dst_index_out. eps is passed as an fp32 bit pattern.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int NUM_REDUCE_TILES>
inline void llk_math_o_norm_sfpu(
    std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_in2, std::uint32_t dst_index_out, std::uint32_t eps_bits) {
    // Start at tile 0: calculate_o_norm converts each operand index to an
    // absolute sfpi::dst_reg[...] offset (the same absolute-index convention as
    // the ternary SFPU params wrapper), so the destination base must not be
    // rebased to dst_index_in0.
    _llk_math_eltwise_sfpu_start_(0);
    sfpu::calculate_o_norm<APPROXIMATION_MODE, is_fp32_dest_acc_en, data_format, NUM_REDUCE_TILES>(
        dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out, eps_bits);
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel
