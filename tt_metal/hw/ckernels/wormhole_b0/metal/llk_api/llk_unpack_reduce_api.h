// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_reduce.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK REDUCE
 *************************************************************************/

/**
 * @brief Initialize the unpacker for a reduce-with-scaler operation.
 *
 * Derives the SrcB (scaler) data formats from the operand's circular buffer, sets haloize
 * (transpose) mode according to the reduce dimension and the transpose flag, programs the datum
 * count, and programs the reduce MOP.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param within_face_16x16_transpose: Nonzero to enable the 16x16 within-face transpose.
 * @ref llk_unpack_reduce is the matching execute call.
 * @ref llk_math_reduce_init is the matching init on the math thread (single-operand unpack pairing).
 */
template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_init(const std::uint32_t within_face_16x16_transpose = 0) {
    constexpr std::uint32_t unpA_operand_id = 0;

    const std::uint32_t unpB_src_format = (std::uint32_t)DataFormat::Float32;
    const std::uint32_t unpB_dst_format =
        ((std::uint32_t)unpack_dst_format[unpA_operand_id] == (std::uint32_t)DataFormat::Int8)
            ? (std::uint32_t)DataFormat::Float16
            :  // Int8 is treated as fp16_a
            ((((std::uint32_t)unpack_dst_format[unpA_operand_id] >> 2) & 0x1) ? (std::uint32_t)DataFormat::Float16_b
                                                                              : (std::uint32_t)DataFormat::Float16);

    _llk_unpack_reduce_init_<type, dim>(unpB_src_format, unpB_dst_format, within_face_16x16_transpose);
}

/**
 * @brief Unpack a tile for a reduce-with-scaler operation.
 *
 * Resolves the tile's L1 address from the operand's circular buffer and tile index, then runs the
 * reduce MOP.
 *
 * @tparam type: Reduction pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param operand: Circular-buffer index of the operand to unpack.
 * @param tile_index: Index of the tile within the circular buffer.
 * @note Call @ref llk_unpack_reduce_init with matching template args before this function.
 */
template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;

    LLK_ASSERT(cb_access_within_bounds(operand_id, tile_index, 1), "Indexed tile read exceeds CB boundary");

    WAYPOINT("UPRW");
    _llk_unpack_reduce_<type, dim>(address);
    WAYPOINT("UPRD");
}
