// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

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
        LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

        _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
            dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    }
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand = 0) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Tilize,
        "Blackhole math datacopy init supports only PackMode::Default and PackMode::Tilize");
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);

    // Post-PR2: FP8 unpack_tilize emits 1 SrcA SetDvalid + 1 SrcB SET_DVALID per tile
    // via the inline 2-context path, matching the non-8-bit whole-tile workaround.
    // Math uses the whole-tile MOP uniformly for all formats — skip_bh_tilize_workaround
    // is false unconditionally.
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(
        num_faces, dst_format, false /* skip_bh_tilize_workaround */);
}

template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {
    _llk_math_eltwise_unary_datacopy_uninit_<src_b_bcast_type, unpack_to_dest>();
}
