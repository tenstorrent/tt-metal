// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_cb_tile_access.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_reduce.h"

/*************************************************************************
 * LLK UNPACK REDUCE
 *************************************************************************/

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

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t address = llk_unpack_tile_address(operand_id, tile_index);

    LLK_ASSERT_BLOCK(validate_unpack_tile_access(operand_id, tile_index, 1));

    WAYPOINT("UPRW");
    _llk_unpack_reduce_<type, dim>(address);
    WAYPOINT("UPRD");
}
