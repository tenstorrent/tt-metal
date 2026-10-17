// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_reduce.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK UNPACK REDUCE
 *************************************************************************/

/**
 * @brief Initialize the unpacker for a single-operand reduce: the data tile goes to SrcA, the scaler tile
 *        (an ordinary circular-buffer tile) to SrcB.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>; REDUCE_ROW is MAX only
 *         (SUM/AVG must use llk_unpack_AB_reduce_init / llk_unpack_AB_reduce).
 * @param scaler_operand: Circular buffer holding the scaler tile; its formats configure the SrcB unpacker.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose (no default: both
 *        arguments are required).
 * @note scaler_operand must also be the SrcB operand passed to hw_startup / reconfig_data_format_srcb (as reduce_init
 *       does via state_configure(icb, icb_scaler, ocb)), and llk_unpack_reduce must use a scaler CB with the same
 *       formats: this init rewrites only the SrcB unpacker in/out formats, while the SrcB signedness bit and the
 *       INT8 math enable stay keyed to that operand, and a later two-operand reconfig_data_format_srcb compares
 *       against that operand's formats and may skip.
 * @note Only the first FACE_C_DIM datums of face 0 of the scaler tile are used, for every face of the data tile;
 *       per-face or partial scaler tiles need llk_unpack_AB_reduce_init / llk_unpack_AB_reduce.
 * @note Pair with @ref llk_unpack_reduce for the per-tile unpack.
 */
template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_init(
    const std::uint32_t scaler_operand, const std::uint32_t within_face_16x16_transpose) {
    SAN_HOOK(unsupported());
    const std::uint32_t scaler_operand_id = get_operand_id(scaler_operand);

    // The SrcB unpacker must already be configured for the scaler CB (by hw_startup / reconfig_data_format_srcb);
    // checked before the init below rewrites the SrcB formats, after which the check would pass trivially.
    LLK_ASSERT_BLOCK((is_unpacker_A_configured_correctly<UnpackerProgramType::ProgramByTile, true /*check_unpacker_b*/>(
        unpack_src_format[scaler_operand_id],
        unpack_dst_format[scaler_operand_id],
        get_operand_face_r_dim(scaler_operand_id),
        get_operand_num_faces(scaler_operand_id))));

    _llk_unpack_reduce_init_<type, dim>(
        unpack_src_format[scaler_operand_id], unpack_dst_format[scaler_operand_id], within_face_16x16_transpose);
}

/**
 * @brief Unpack one data tile into SrcA and the scaler tile into SrcB for a single-operand reduce.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>; REDUCE_ROW is MAX only (see
 *         llk_unpack_reduce_init).
 * @param operand: Circular buffer holding the data tile.
 * @param scaler_operand: Circular buffer holding the scaler tile (only face 0, row 0 of it is used, for every
 *        face of the data tile).
 * @param tile_index: Index of the data tile within its circular buffer.
 * @param scaler_tile_index: Index of the scaler tile within its circular buffer.
 * @note Call @ref llk_unpack_reduce_init with matching template args before this function.
 * @note 32x32 tiles only: the unpack MOP always issues four 16-row face unpacks per tile, while the math takes
 *       the face count from the data CB, so a 2-face or tiny-tile data CB would misalign the two threads.
 */
template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce(
    const std::uint32_t operand,
    const std::uint32_t scaler_operand,
    const std::uint32_t tile_index,
    const std::uint32_t scaler_tile_index = 0) {
    SAN_HOOK(unsupported());
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t scaler_operand_id = get_operand_id(scaler_operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;
    std::uint32_t scaler_base_address = get_local_cb_interface(scaler_operand_id).fifo_rd_ptr - 1;
    std::uint32_t scaler_offset_address = get_local_cb_interface(scaler_operand_id).fifo_page_size * scaler_tile_index;
    std::uint32_t scaler_address = scaler_base_address + scaler_offset_address;

    LLK_ASSERT(
        cb_access_within_bounds(operand_id, tile_index, 1 /*num_tiles*/), "Indexed tile read exceeds CB boundary");
    LLK_ASSERT(
        get_operand_num_faces(operand_id) == TILE_NUM_FACES && get_operand_face_r_dim(operand_id) == FACE_R_DIM,
        "Single-operand reduce supports 32x32 tiles only");

    // llk_unpack_reduce_init programs SrcB only; SrcA must hold the data CB's formats from hw_startup / reconfig.
    LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        unpack_src_format[scaler_operand_id],
        unpack_dst_format[scaler_operand_id],
        get_operand_face_r_dim(operand_id),
        get_operand_face_r_dim(scaler_operand_id),
        get_operand_num_faces(operand_id),
        get_operand_num_faces(scaler_operand_id)));
    LLK_ASSERT(
        cb_access_within_bounds(scaler_operand_id, scaler_tile_index, 1 /*num_tiles*/),
        "Indexed tile read exceeds CB boundary");

    WAYPOINT("UPRW");
    _llk_unpack_reduce_<type, dim>(address, scaler_address);
    WAYPOINT("UPRD");
}
