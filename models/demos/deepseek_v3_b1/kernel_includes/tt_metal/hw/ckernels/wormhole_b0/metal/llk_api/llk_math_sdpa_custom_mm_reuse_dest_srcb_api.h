// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "../../../../../third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_sdpa_custom_mm_reuse_dest_srcb.h"

template <MathFidelity math_fidelity>
inline void llk_math_sdpa_custom_mm_reuse_dest_srcb_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t kt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const bool partial_face = 0;

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_<math_fidelity>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face, transpose, kt_dim);
}

template <MathFidelity math_fidelity>
inline void llk_math_sdpa_custom_mm_reuse_dest_srcb(
    const uint src_index,
    const uint dst_index,
    const bool transpose = false,
    const std::uint32_t kt_dim = 1,
    const std::uint32_t nt_dim = 1,
    bool signal_output = false) {
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_(src_index, dst_index, transpose, kt_dim, nt_dim, signal_output);
}
