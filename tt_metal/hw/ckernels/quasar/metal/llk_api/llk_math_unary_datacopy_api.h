// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
 * For an unpack-to-dest operand the unpacker writes DEST directly: on the plain path math has no MOP to
 * program and the math init is skipped; on a broadcast path math broadcasts the placed rows in place
 * (MOVD2B / MOVB2D), which needs the mov-ops ALU format state instead of the default one.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>; any value with unpack_to_dest
 * @tparam EN_32BIT_DEST set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast mode; non-NONE uses the unary-broadcast math init
 * @tparam unpack_to_dest Set true to enable the UNP_DEST routing decision (unary and unary-broadcast paths)
 * @param operand Logical dataflow buffer id for the input operand
 */
template <
    DataCopyType type,
    bool EN_32BIT_DEST,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false,
    bool is_int_fpu_en /*maybe_unused*/ = false,
    bool tilize /*maybe_unused*/ = false>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_rows = num_faces * face_r_dim;

    const DataFormat srcA_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));
    const DataFormat srcB_format = static_cast<DataFormat>(get_operand_dst_format(operand_id));

    if constexpr (src_b_bcast_type == BroadcastType::NONE) {
        _configure_default_alu_data_format_state_<EN_32BIT_DEST>(srcA_format, srcB_format);
        // 32-bit unpack-to-dest: math is a sync-only forwarder (unpacker wrrites DEST), no MOP to run.
        if constexpr (!unpack_to_dest) {
            _llk_math_eltwise_unary_datacopy_init_<type, EN_32BIT_DEST>(
                num_rows /*num_rows_per_matrix*/, 1 /*num_matrices*/);
        }
    } else {
        static_assert(unpack_to_dest || type == DataCopyType::B2D, "Unary broadcast from SrcB requires B2D");

        const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand);
        LLK_ASSERT(
            tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM &&
                tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
            "Unary broadcast currently only supports 32x32 tiles (face_r_dim=16, 2x2 faces)");

        if constexpr (unpack_to_dest) {
            // DEST -> SrcB -> DEST moves need the mov-ops ALU state (Tf32 sources, Float32 dest override on 32-bit).
            _configure_mov_ops_explicit_alu_data_format_state_<EN_32BIT_DEST>(srcA_format, srcB_format);
        } else {
            _configure_default_alu_data_format_state_<EN_32BIT_DEST>(srcA_format, srcB_format);
        }
        _llk_math_eltwise_unary_broadcast_init_<src_b_bcast_type, EN_32BIT_DEST, unpack_to_dest>(tensor_shape);
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a single tile.
 *
 * For the non-broadcast path, when `unpack_to_dest` is true the unpacker writes DEST directly and
 * math is a sync-only forwarder; no MOP is run. A broadcast operand in DEST is broadcast in place.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>; any value with unpack_to_dest
 * @tparam EN_32BIT_DEST set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast mode; non-NONE runs the unary-broadcast math (from SrcB or from DEST)
 * @tparam unpack_to_dest when true, unpack-to-dest path; SrcA / SrcB path otherwise
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

    if constexpr (src_b_bcast_type != BroadcastType::NONE) {
        static_assert(unpack_to_dest || type == DataCopyType::B2D, "Unary broadcast from SrcB requires B2D");
        LLK_ASSERT(
            (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()),
            "llk_math_eltwise_unary_datacopy: dst_index exceeds dest capacity");
        _llk_math_eltwise_unary_broadcast_<unpack_to_dest>(dst_index);
    } else {
        // 32-bit unpack-to-dest: math is a sync-only forwarder (unpacker wrrites DEST), no MOP to run.
        if constexpr (!unpack_to_dest) {
            LLK_ASSERT(
                (dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE>()),
                "llk_math_eltwise_unary_datacopy: dst_index exceeds dest capacity");
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
        LLK_ASSERT(
            (start_dst_index + ntiles <= get_dest_max_tiles<DST_SYNC_MODE, EN_32BIT_DEST>()),
            "llk_math_eltwise_unary_datacopy_block: dst range exceeds dest capacity");
        for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
            _llk_math_eltwise_unary_datacopy_(dst_index);
        }
    }
}

template <
    BroadcastType src_b_bcast_type /*maybe_unused*/ = BroadcastType::NONE,
    bool unpack_to_dest /*maybe_unused*/ = false>
inline void llk_math_eltwise_unary_datacopy_uninit() {}
