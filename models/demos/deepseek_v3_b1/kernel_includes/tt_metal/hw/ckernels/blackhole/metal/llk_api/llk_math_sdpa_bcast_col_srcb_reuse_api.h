// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "../../../../../../tt_llk/tt_llk_blackhole/llk_lib/llk_math_sdpa_bcast_col_srcb_reuse.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with operands
template <EltwiseBinaryType eltwise_binary_type, uint32_t num_tiles, int NUM_FIDELITY_PHASES = 0>
inline void llk_math_sdpa_bcast_col_srcb_reuse_init_with_operands(
    const std::uint32_t operand_A, const std::uint32_t operand_B, const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_sdpa_bcast_col_srcb_reuse_init_<eltwise_binary_type, num_tiles, NUM_FIDELITY_PHASES>(
        num_faces, acc_to_dest);
}

template <DstSync DST, bool IS_FP32_MATH_DEST_EN, bool clear_dest = false>
inline void llk_math_sdpa_bcast_col_srcb_reuse_preamble() {
    _llk_math_sdpa_bcast_col_srcb_reuse_preamble_<DST, IS_FP32_MATH_DEST_EN, clear_dest>();
}

inline void llk_math_sdpa_bcast_col_srcb_reuse_postamble() { _llk_math_sdpa_bcast_col_srcb_reuse_postamble_(); }

template <
    EltwiseBinaryType eltwise_binary_type,
    uint32_t num_tiles,
    bool is_fp32_dest_acc_en,
    int NUM_FIDELITY_PHASES = 0>
inline void llk_math_sdpa_bcast_col_srcb_reuse(const std::uint32_t dst_index) {
    _llk_math_sdpa_bcast_col_srcb_reuse_<
        eltwise_binary_type,
        num_tiles,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        NUM_FIDELITY_PHASES>(dst_index);
}
