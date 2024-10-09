// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

template <
    DataCopyType type,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_fp32_dest_acc_en = false,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(uint dst_index, uint operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, src_b_bcast_type, is_fp32_dest_acc_en, unpack_to_dest>(
        dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
}

template <
    DataCopyType type,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_fp32_dest_acc_en = false,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(uint start_dst_index, uint ntiles, uint operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    for (uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, src_b_bcast_type, is_fp32_dest_acc_en, unpack_to_dest>(
            dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    }
}

template <DataCopyType type, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool is_fp32_dest_acc_en = false, bool is_int_fpu_en = false, bool tilize = false/*unused*/>
// within_face_16x16_transpose is used by unpacker, math does not transpose
inline void llk_math_eltwise_unary_datacopy_init(
    const std::uint32_t transpose_of_faces = 0 /*unused*/,
    const std::uint32_t within_face_16x16_transpose = 0 /* unused */,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    _llk_math_eltwise_unary_datacopy_init_<type, src_b_bcast_type, is_fp32_dest_acc_en, is_int_fpu_en>(
        transpose_of_faces, within_face_16x16_transpose, num_faces, dst_format);
}
