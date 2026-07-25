// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_unary_broadcast.h"
#include "llk_operands.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 * @brief Initialize eltwise unary datacopy operations
 *
 * For an unpack-to-dest operand the unpacker writes DEST directly and math has no MOP to
 * program; that case is gated on the `unpack_to_dest` template parameter and the math init is skipped.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam EN_32BIT_DEST set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast mode; non-NONE uses unary-broadcast math init when type is B2D
 * @tparam unpack_to_dest Set true to enable the UNP_DEST routing decision (unary, non-broadcast path)
 * @param operand Logical dataflow buffer id for the input operand
 */
template <
    DataCopyType type,
    bool EN_32BIT_DEST,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false,
    [[maybe_unused]] bool is_int_fpu_en = false,
    [[maybe_unused]] bool tilize = false>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_rows = num_faces * face_r_dim;

    const DataFormat srcA_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));
    const DataFormat srcB_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));
    _configure_default_alu_data_format_state_<false /* IMPLIED_MATH_FORMAT */, EN_32BIT_DEST>(srcA_format, srcB_format);

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        // 32-bit unpack-to-dest: math is a sync-only forwarder (unpacker wrrites DEST), no MOP to run.
        if constexpr (!unpack_to_dest) {
            _llk_math_eltwise_unary_datacopy_init_<type, EN_32BIT_DEST>(
                num_rows /*num_rows_per_matrix*/, 1 /*num_matrices*/);
        }
    } else {
        static_assert(type == DataCopyType::B2D);
        const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand);
        _llk_math_eltwise_unary_broadcast_init_<src_b_bcast_type, false /*unpack_to_dest*/, EN_32BIT_DEST>(
            tensor_shape);
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a single tile.
 *
 * For the non-broadcast path, when `unpack_to_dest` is true the unpacker writes DEST directly and
 * math is a sync-only forwarder; no MOP is run.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam EN_32BIT_DEST set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast mode; non-NONE with unpack_to_dest false uses unary-broadcast math
 * @tparam unpack_to_dest when true, unpack-to-dest path; plain datacopy otherwise
 * @param dst_index Tile index into the destination register.
 * @param operand Logical dataflow buffer id for the input operand.
 */
template <
    DataCopyType type = DataCopyType::A2D,
    bool EN_32BIT_DEST = false,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);

    if constexpr (src_b_bcast_type != BroadcastType::NONE && !unpack_to_dest) {
        static_assert(type == DataCopyType::B2D, "Unary broadcast math path requires DataCopyType::B2D");
        const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand);
        _llk_math_eltwise_unary_broadcast_<src_b_bcast_type, false, EN_32BIT_DEST>(dst_index, tensor_shape);
    } else {
        // 32-bit unpack-to-dest: math is a sync-only forwarder (unpacker wrrites DEST), no MOP to run.
        if constexpr (!unpack_to_dest) {
            _llk_math_eltwise_unary_datacopy_(dst_index);
        }
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a block of tiles.
 *
 * When `unpack_to_dest` is true the unpacker writes DEST directly and math is a sync-only forwarder;
 * no MOP is run.
 *
 * @param start_dst_index Starting tile index in the destination register.
 * @param ntiles Number of tiles to copy to the destination register.
 * @param operand Logical dataflow buffer id for the input operand.
 */
template <
    DataCopyType type = DataCopyType::A2D,
    bool EN_32BIT_DEST = false,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    const std::uint32_t start_dst_index, const std::uint32_t ntiles, const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_rows = num_faces * face_r_dim;

    // 32-bit unpack-to-dest: math is a sync-only forwarder (unpacker wrrites DEST), no MOP to run.
    if constexpr (!unpack_to_dest) {
        for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
            _llk_math_eltwise_unary_datacopy_(dst_index);
        }
    }
}

template <
    [[maybe_unused]] BroadcastType src_b_bcast_type = BroadcastType::NONE,
    [[maybe_unused]] bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {}
