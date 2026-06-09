// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_untilize.h"

/*************************************************************************
 * LLK UNPACK UNTILIZE
 *************************************************************************/

/**
 * @brief Initialize the unpacker for an untilize operation.
 *
 * Programs the unpacker for untilize, deriving the destination format and tile size from the
 * operand's circular buffer.
 *
 * @param operand: Circular-buffer index of the operand to untilize.
 * @note Call @ref llk_unpack_untilize_uninit to restore the default unpacker config.
 * @ref llk_math_eltwise_unary_datacopy_init (A2D) is the matching init on the math thread.
 */
inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;

    _llk_unpack_untilize_init_(
        unpack_dst_format[operand_id], get_local_cb_interface(operand_id).fifo_page_size, face_r_dim);
}

/**
 * @brief Restore unpacker state after an untilize operation.
 *
 * Derives the destination format from the operand's circular buffer.
 *
 * @param operand: Circular-buffer index of the operand that was untilized.
 * @param face_r_dim: Rows per face, used to compute the restored datum count and Y stride.
 * @note Call @ref llk_unpack_untilize_init before this function.
 */
inline void llk_unpack_untilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    std::uint32_t operand_id = get_operand_id(operand);
    WAYPOINT("UPUW");
    _llk_unpack_untilize_uninit_((uint)unpack_dst_format[operand_id], face_r_dim);
    WAYPOINT("UPUD");
}

/**
 * @brief Run one untilize pass (top or bottom faces) over a row of tiles.
 *
 * Resolves the tile-row base address from the operand's circular buffer.
 *
 * @tparam first_pass: Select the top faces (true) or bottom faces (false) of each tile.
 * @param operand: Circular-buffer index of the operand to untilize.
 * @param block_tile_cols: Number of tile columns in the block row.
 * @note Call @ref llk_unpack_untilize_init before this function, and
 *       @ref llk_unpack_untilize_uninit after the final pass to restore modified state.
 */
template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;

    _llk_unpack_untilize_pass_<first_pass>(base_address, block_tile_cols);
}

/**
 * @brief Untilize a row of tiles by running both unpack passes.
 *
 * Runs @ref llk_unpack_untilize_pass for the top faces then the bottom faces of the row.
 *
 * @param operand: Circular-buffer index of the operand to untilize.
 * @param block_c_tiles: Number of tile columns in the block row.
 * @note Call @ref llk_unpack_untilize_init before this function, and
 *       @ref llk_unpack_untilize_uninit after to restore modified state.
 */
inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    WAYPOINT("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    WAYPOINT("UPUD");
}
