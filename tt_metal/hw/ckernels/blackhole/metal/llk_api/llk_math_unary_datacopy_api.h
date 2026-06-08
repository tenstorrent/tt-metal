// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 * @brief Copy a tile into the destination register, optionally broadcasting source B.
 *
 * Derives the source and destination data formats from the operand's circular buffer.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Unpack writes directly to dest (vs. via source registers).
 * @param dst_index: Tile index into the destination register.
 * @param operand: Circular-buffer index of the operand to copy.
 * @note Call @ref llk_math_eltwise_unary_datacopy_init with matching template args before this function, and
 *       @ref llk_math_eltwise_unary_datacopy_uninit after it to restore modified state.
 * @note On the unpack thread, @ref llk_unpack_A must feed the tile into SrcA/SrcB (or dest for unpack-to-dest).
 */
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

/**
 * @brief Copy a contiguous block of tiles into the destination register.
 *
 * Loops @ref llk_math_eltwise_unary_datacopy over @p ntiles destination tiles starting at
 * @p start_dst_index, deriving the data formats from the operand's circular buffer.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Unpack writes directly to dest (vs. via source registers).
 * @param start_dst_index: First tile index into the destination register.
 * @param ntiles: Number of consecutive tiles to copy.
 * @param operand: Circular-buffer index of the operand to copy.
 * @note Call @ref llk_math_eltwise_unary_datacopy_init with matching template args before this function, and
 *       @ref llk_math_eltwise_unary_datacopy_uninit after it to restore modified state.
 */
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

/**
 * @brief Initialize the math thread (address mods and MOP) for an elementwise unary datacopy.
 *
 * Derives num_faces and the source/destination formats from the operand's circular buffer.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam is_int_fpu_en: Enable integer FPU datapath.
 * @tparam pack_mode: Packing layout, values = <Default/Tilize>
 * @param operand: Circular-buffer index of the operand to copy.
 * @note On the unpack thread, pair with @ref llk_unpack_A_init (copy/transpose),
 *       @ref llk_unpack_tilize_init (tilize) or @ref llk_unpack_untilize_init.
 * @note @ref llk_math_eltwise_unary_datacopy runs the configured op with matching template args.
 */
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

    // For tilize operation, the init function needs to know the src format to determine the is_8bit_format to avoid the
    // tilize workaround. 8bit datums in input format do not require the tilize workaround on blackhole.
    const std::uint32_t src_format = get_operand_src_format(operand_id);
    const bool is_input_8bit_format = IS_8BIT_FORMAT(src_format);
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(
        num_faces, dst_format, is_input_8bit_format);
}

/**
 * @brief Uninitialize after an elementwise unary datacopy, undoing init-time workarounds.
 *
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Whether unpack wrote directly to dest.
 * @note Reverses @ref llk_math_eltwise_unary_datacopy_init.
 */
template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {
    _llk_math_eltwise_unary_datacopy_uninit_<src_b_bcast_type, unpack_to_dest>();
}
