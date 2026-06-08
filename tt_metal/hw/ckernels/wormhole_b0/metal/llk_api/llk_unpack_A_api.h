// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_A.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

/**
 * @brief Initialize the unpacker for a single-operand (A) unpack.
 *
 * Derives face_r_dim, num_faces, and the data formats from the operand's circular buffer, then
 * programs the within-face transpose register, datum count, and MOP for the requested mode.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param transpose_of_faces: Nonzero to reorder (transpose) faces during the unpack.
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose (haloize mode).
 * @param operand: Circular-buffer index of the operand to unpack.
 * @note Call @ref llk_unpack_A_uninit after unpacking to restore the modified datum-count state.
 * @ref llk_unpack_A is the matching execute call.
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    const std::uint32_t operand_unpack_src_format = unpack_src_format[operand_id];
    const std::uint32_t operand_unpack_dst_format = unpack_dst_format[operand_id];

    LLK_ASSERT_BLOCK((is_unpacker_A_configured_correctly<
                      UnpackerProgramType::ProgramByTile,
                      (BType != BroadcastType::NONE && !unpack_to_dest)>(
        operand_unpack_src_format, operand_unpack_dst_format, face_r_dim, num_faces)));

    _llk_unpack_A_init_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces,
        within_face_16x16_transpose,
        face_r_dim,
        num_faces,
        operand_unpack_src_format,
        operand_unpack_dst_format);
}

/**
 * @brief Unpack a single tile (operand A) from L1 into the SrcA/SrcB or dest register.
 *
 * Resolves the tile's L1 address from the operand's circular buffer and the given tile index,
 * then runs the configured MOP.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param operand: Circular-buffer index of the operand to unpack.
 * @param tile_index: Index of the tile within the circular buffer.
 * @note Call @ref llk_unpack_A_init with matching template args before this function, and
 *       @ref llk_unpack_A_uninit after it to restore modified state.
 * @ref llk_math_eltwise_unary_datacopy on the math thread consumes the tile unpacked here.
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;

    LLK_ASSERT(cb_access_within_bounds(operand_id, tile_index, 1), "Indexed tile read exceeds CB boundary");

    LLK_ASSERT_BLOCK((is_unpacker_A_configured_correctly<
                      UnpackerProgramType::ProgramByTile,
                      (BType != BroadcastType::NONE && !unpack_to_dest)>(
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        get_operand_face_r_dim(operand_id),
        get_operand_num_faces(operand_id))));

    WAYPOINT("UPAW");
    _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    WAYPOINT("UPAD");
}

/**
 * @brief Unpack a contiguous block of tiles (operand A) from L1.
 *
 * Resolves the starting tile's L1 address from the operand's circular buffer, then unpacks
 * @p ntiles consecutive tiles by repeating the single-tile unpack.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @tparam acc_to_dest: Accumulate the operand into the dest register rather than overwriting it.
 * @tparam binary_reuse_dest: Reuse dest as a source operand, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam unpack_to_dest: Unpack directly into the dest register (32-bit datums).
 * @param operand: Circular-buffer index of the operand to unpack.
 * @param start_tile_index: Index of the first tile within the circular buffer.
 * @param ntiles: Number of consecutive tiles to unpack.
 * @note Call @ref llk_unpack_A_init with matching template args before this function, and
 *       @ref llk_unpack_A_uninit after it to restore modified state.
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size;
    std::uint32_t address = base_address + start_tile_index * offset_address;

    LLK_ASSERT(cb_access_within_bounds(operand_id, start_tile_index, ntiles), "Block tile read exceeds CB boundary");

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
            address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
        address += offset_address;
        WAYPOINT("UPAD");
    }
}

/**
 * @brief Restore unpacker datum-count state after single-operand (A) unpacking.
 *
 * Resets the X-dimension address counter back to a full face worth of datums, using the
 * operand's face_r_dim taken from its circular buffer.
 *
 * @tparam BType: Broadcast type, values = <NONE/COL/ROW/SCALAR>
 * @param operand: Circular-buffer index of the operand that was unpacked.
 * @note Call @ref llk_unpack_A_init with matching template args before this function.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_A_uninit(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);

    _llk_unpack_A_uninit_<BType>(face_r_dim);
}
