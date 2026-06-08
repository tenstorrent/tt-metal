// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_fast_tilize.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 * @brief Copy a tile into the destination register, optionally broadcasting source B.
 *
 * Derives the source/destination data formats from the operand's circular buffer.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Unpack writes directly to dest (vs. via source registers).
 * @param dst_index: Tile index into the destination register.
 * @param operand: Circular-buffer index whose data formats are used.
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
 * Repeats the single-tile datacopy over @p ntiles consecutive dest indices, using the data formats
 * derived from the operand's circular buffer.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Unpack writes directly to dest (vs. via source registers).
 * @param start_dst_index: First tile index into the destination register.
 * @param ntiles: Number of consecutive tiles to copy.
 * @param operand: Circular-buffer index whose data formats are used.
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
 * Derives the face count and destination format from the operand's circular buffer.
 *
 * @tparam type: Datacopy direction, values = <A2D/B2D>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam is_int_fpu_en: Enable integer FPU datapath.
 * @tparam pack_mode: Pack mode, values = <Default/Untilize/Tilize> (Tilize is ignored on Wormhole).
 * @param operand: Circular-buffer index whose face count and data format are used.
 * @note On the unpack thread, pair with @ref llk_unpack_A_init which feeds the tile.
 * @ref llk_math_eltwise_unary_datacopy runs the configured op with matching template args.
 */
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand = 0) {
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize || pack_mode == PackMode::Tilize,
        "Wormhole B0 math datacopy init: use PackMode::Default, PackMode::Untilize, or PackMode::Tilize (tilize is "
        "ignored on WH)");
    (void)pack_mode;
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en>(
        num_faces, dst_format);
}

/**
 * @brief Uninitialize after an elementwise unary datacopy.
 *
 * @tparam src_b_bcast_type: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @tparam unpack_to_dest: Whether the datacopy unpacked directly to dest.
 * @note Reverses @ref llk_math_eltwise_unary_datacopy_init.
 */
template <BroadcastType src_b_bcast_type = BroadcastType::NONE, bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {
    _llk_math_eltwise_unary_datacopy_uninit_<src_b_bcast_type, unpack_to_dest>();
}

/*************************************************************************
 * LLK FAST ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 * @brief Initialize the math thread for fast tilize: programs address mods and the move MOP.
 *
 * Takes the destination format from the operand's circular buffer. Only DstSync::SyncHalf is supported.
 *
 * @param operand: Circular-buffer index whose destination format is used.
 * @param unit_dim: Number of tiles processed per iteration; must match the unpacker.
 * @note On the unpack thread, pair with @ref llk_unpack_fast_tilize_init which feeds the top/bottom faces into
 * SrcA/SrcB.
 * @note On the pack thread, pair with @ref llk_pack_fast_tilize_init (same unit_dim) which drains the split dest
 * halves.
 * @note Call @ref llk_math_fast_tilize_uninit to restore the changed state; run with @ref llk_math_fast_tilize_block_.
 */
inline void llk_math_fast_tilize_init(const std::uint32_t operand, const std::uint32_t unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_init_(unpack_dst_format[operand_id], unit_dim);
}

/**
 * @brief Uninitialize after fast tilize, restoring the FP32 dest-accumulation mode and CFG state that init changed.
 *
 * Takes the destination format from the operand's circular buffer; only non-TF32 needs restoring.
 *
 * @tparam is_fp32_dest_acc_en: FP32 dest-accumulation mode to restore (must match the surrounding context).
 * @param operand: Circular-buffer index whose destination format is used.
 * @note Reverses @ref llk_math_fast_tilize_init.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_uninit(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}

/**
 * @brief Tilize a block of tiles into the destination register, moving top and bottom faces into split dest halves.
 *
 * Takes the destination format and face count from the operand's circular buffer. Only DstSync::SyncHalf
 * is supported, and nothing else should use the active dest bank.
 *
 * @param dst_index: Tile index into the destination register to write to.
 * @param operand: Circular-buffer index whose destination format and face count are used.
 * @param unit_dim: Number of tiles processed per iteration; must match the unpacker.
 * @param num_units: Number of units processed in this call.
 * @note Call @ref llk_math_fast_tilize_init with matching operand and unit_dim before this function.
 * @note On the unpack thread, @ref llk_unpack_fast_tilize_block must feed the tiles into SrcA/SrcB.
 * @note On the pack thread, @ref llk_pack_fast_tilize_block drains the split dest halves into tilized L1 output.
 */
inline void llk_math_fast_tilize_block_(
    const std::uint32_t dst_index,
    const std::uint32_t operand,
    const std::uint32_t unit_dim,
    const std::uint32_t num_units) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_fast_tilize_block_(dst_index, unpack_dst_format[operand_id], unit_dim, num_units, num_faces);
}
