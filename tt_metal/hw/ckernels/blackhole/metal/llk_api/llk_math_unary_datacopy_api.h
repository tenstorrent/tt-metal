// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 * Complete the math-side unpack-to-destination handshake for one 32x16 tile.
 *
 * @param dst_index Narrow destination index in [0, get_dest_max_tiles_rt(..., Tile32x16)).
 * @param operand Logical input operand; its unpack source and destination formats must both be 32-bit.
 */
__attribute__((always_inline)) inline void llk_math_eltwise_unary_datacopy_to_dest_32x16(
    std::uint32_t dst_index, std::uint32_t operand) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x16>()), "");
    const std::uint32_t operand_id = get_operand_id(operand);
    const bool unpack_to_dest = is_32bit_input(unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    LLK_ASSERT(unpack_to_dest, "32x16 unpack-to-dest requires a 32-bit operand format");
    if (!unpack_to_dest) {
        return;
    }
#ifdef SAN_HOOK
    SAN_HOOK(unsupported());
#endif
    math::math_unpack_to_dest_math_ready();
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::DestReg>(dst_index);
    math::math_unpack_to_dest_tile_ready();
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(std::uint32_t dst_index, std::uint32_t operand) {
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_eltwise_unary_datacopy_<type, DST_SYNC_MODE, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
        dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    std::uint32_t start_dst_index, std::uint32_t ntiles, std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);

    for (uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        LLK_ASSERT((dst_index < get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()), "");

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
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Tilize,
        "Blackhole math datacopy init supports only PackMode::Default and PackMode::Tilize");
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);

    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(
        num_faces, dst_format);
}

template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {
    _llk_math_eltwise_unary_datacopy_uninit_<src_b_bcast_type, unpack_to_dest>();
}
