// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../../../../../tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_A_rmsnorm.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

template <
    uint32_t num_tiles,
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_rmsnorm_init(
    const std::uint32_t transpose_of_faces = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    const std::uint32_t operand_unpack_src_format = unpack_src_format[operand_id];
    const std::uint32_t operand_unpack_dst_format = unpack_dst_format[operand_id];
    if (unpack_to_dest && is_32bit_input(operand_unpack_src_format, operand_unpack_dst_format)) {
        llk_unpack_dbg_feature_disable();
    }

    _llk_unpack_A_rmsnorm_init_<num_tiles, BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces,
        within_face_16x16_transpose,
        face_r_dim,
        num_faces,
        operand_unpack_src_format,
        operand_unpack_dst_format);
}
