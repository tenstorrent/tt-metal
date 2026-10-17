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
 * Single-operand reduce: the data tile goes to SrcA, the scaler tile (an ordinary tile in a circular
 * buffer) to SrcB. Formats for SrcB come from the scaler operand's CB configuration.
 */
template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_init(
    const std::uint32_t scaler_operand, const std::uint32_t within_face_16x16_transpose = 0) {
    SAN_HOOK(unsupported());
    const std::uint32_t scaler_operand_id = get_operand_id(scaler_operand);

    _llk_unpack_reduce_init_<type, dim>(
        unpack_src_format[scaler_operand_id], unpack_dst_format[scaler_operand_id], within_face_16x16_transpose);
}

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

    LLK_ASSERT(cb_access_within_bounds(operand_id, tile_index, 1), "Indexed tile read exceeds CB boundary");
    LLK_ASSERT(
        cb_access_within_bounds(scaler_operand_id, scaler_tile_index, 1), "Indexed tile read exceeds CB boundary");

    WAYPOINT("UPRW");
    _llk_unpack_reduce_<type, dim>(address, scaler_address);
    WAYPOINT("UPRD");
}
