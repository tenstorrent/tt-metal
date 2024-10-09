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
inline void llk_math_eltwise_unary_datacopy(uint dst_index, uint operand = 0 /* unused */) {

    _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, src_b_bcast_type, is_fp32_dest_acc_en>(dst_index);
}

template <DataCopyType type, BroadcastType src_b_bcast_type = BroadcastType::NONE,  bool is_fp32_dest_acc_en = false, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(uint start_dst_index, uint ntiles, uint operand = 0 /*not used*/) {

    for (uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        llk_math_eltwise_unary_datacopy<type, src_b_bcast_type>(dst_index);
    }
}

template <DataCopyType type, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool is_fp32_dest_acc_en = false/*unused*/, bool is_int_fpu_en = false/*unused*/, bool tilize = false/*unused*/>
// within_face_16x16_transpose is used by unpacker, math does not transpose
inline void llk_math_eltwise_unary_datacopy_init(
    const std::uint32_t transpose_of_faces = 0 /*unused*/,
    const std::uint32_t within_face_16x16_transpose = 0 /* unused */,
    const std::uint32_t operand = 0 /* unused */) {
    _llk_math_eltwise_unary_datacopy_init_<type, src_b_bcast_type>(transpose_of_faces, within_face_16x16_transpose);
}
