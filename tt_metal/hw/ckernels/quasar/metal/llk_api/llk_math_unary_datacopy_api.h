// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_unary_broadcast_api.h"
#include "llk_operands.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 *
 * @brief Initialize eltwise unary datacopy operations
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam is_fp32_dest_acc_en set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast mode; non-NONE uses unary-broadcast math init when type is B2D
 * @tparam is_int_fpu_en Same template slot as Blackhole; unused on Quasar (Quasar _llk init has no equivalent).
 * @tparam tilize Same template slot as Blackhole; unused on Quasar.
 * @param operand Logical dataflow buffer id for the input operand
 */
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    [[maybe_unused]] bool is_int_fpu_en = false,
    [[maybe_unused]] bool tilize = false>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_rows = num_faces * face_r_dim;

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _llk_math_eltwise_unary_datacopy_init_<type, is_fp32_dest_acc_en>(
            num_rows /*num_rows_per_matrix*/, 1 /*num_matrices*/);
    } else if constexpr (type == DataCopyType::B2D) {
        llk_math_eltwise_unary_broadcast_init<src_b_bcast_type, false, is_fp32_dest_acc_en>(operand);
    } else {
        static_assert(type == DataCopyType::A2D);
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
            num_rows /*num_rows_per_matrix*/, 1 /*num_matrices*/);
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a single tile.
 *
 * @param dst_index Tile index into the destination register.
 * @param operand The input operand logical dataflow buffer id.
 *
 * @param dst_index: Tile index into the destination register
 * @param operand: The input operand circular buffer
 *
 * This function copies a specified number of rows
 * from the srcA or srcB register to the destination register.
 */
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand = 0) {
    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        const std::uint32_t operand_id = get_operand_id(operand);
        const std::uint32_t num_faces = get_operand_num_faces(operand_id);
        const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
        _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
    } else if constexpr (!unpack_to_dest) {
        llk_math_eltwise_unary_broadcast<src_b_bcast_type, false, is_fp32_dest_acc_en>(dst_index, operand);
    } else {
        const std::uint32_t operand_id = get_operand_id(operand);
        const std::uint32_t num_faces = get_operand_num_faces(operand_id);
        const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
        _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a block of tiles.
 *
 * @param start_dst_index Starting tile index in the destination register.
 * @param ntiles Number of tiles to copy to the destination register.
 * @param operand The input operand logical dataflow buffer id.
 *
 * This function copies a contiguous block of tiles
 * from the srcA or srcB register to the destination register.
 */
inline void llk_math_eltwise_unary_datacopy_block(
    const std::uint32_t start_dst_index, const std::uint32_t ntiles, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);

    for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
    }
}

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    const std::uint32_t start_dst_index, const std::uint32_t ntiles, const std::uint32_t operand = 0) {
    for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
        llk_math_eltwise_unary_datacopy<type, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
            dst_index, operand);
    }
}

template <
    [[maybe_unused]] BroadcastType src_b_bcast_type = BroadcastType::NONE,
    [[maybe_unused]] bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {}
