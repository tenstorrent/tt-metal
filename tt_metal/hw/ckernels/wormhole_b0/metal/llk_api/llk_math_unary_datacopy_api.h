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
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(uint dst_index, uint operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
        dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(uint start_dst_index, uint ntiles, uint operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    for (uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
            dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    }
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    bool tilize = false /*unused*/>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en>(
        num_faces, dst_format);
}

template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {
    _llk_math_eltwise_unary_datacopy_uninit_<src_b_bcast_type, unpack_to_dest>();
}

/*************************************************************************
 * LLK FAST ELTWISE UNARY DATACOPY
 *************************************************************************/

inline void llk_math_fast_tilize_init(const std::uint32_t operand, const std::uint32_t unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_init_(unpack_dst_format[operand_id], unit_dim);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_uninit(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}

inline void llk_math_fast_tilize_block_(
    const std::uint32_t dst_index,
    const std::uint32_t operand,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_block_(dst_index, unpack_dst_format[operand_id], unit_dim, num_units);
}
