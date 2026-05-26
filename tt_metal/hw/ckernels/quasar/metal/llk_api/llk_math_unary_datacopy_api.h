// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_operands.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

/**
 *
 * @brief Initialize eltwise unary datacopy operations
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam IS_32b_DEST_EN set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast type; carried for API parity with non-Quasar architectures.
 * @tparam unpack_to_dest When true, the caller has configured the unpacker to write directly to
 *     dest (UNPACR_DEST); the math thread skips its MOV/ELWADD MOP and simply waits for the
 *     unpacker to release each DEST section. Plumbed for API parity with non-Quasar; the actual
 *     wait happens in the compute API (tile_regs_acquire/commit) and inside the unpack API.
 * @param operand: The input operand circular buffer
 * This function prepares the math hardware to copy a specified number of rows
 * from the srcA or srcB register to the destination register.
 */
template <
    DataCopyType type,
    bool IS_32b_DEST_EN,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    static_assert(
        src_b_bcast_type == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    _llk_math_eltwise_unary_datacopy_init_<type, IS_32b_DEST_EN>(
        num_faces * face_r_dim /*num_rows_per_matrix*/, 1 /*num_matrices*/);
}

/**
 * @brief Performs an eltwise unary datacopy for a single tile.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam IS_32b_DEST_EN set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast type; carried for API parity with non-Quasar architectures.
 * @tparam unpack_to_dest When true, the unpacker drives dest directly via UNPACR_DEST and the
 *     math thread does not issue the MOV/ELWADD MOP for this tile (the tile is already in dest).
 *     The UNPACK_MATH semaphore handshake is performed at the compute-API layer
 *     (tile_regs_acquire/commit), not here.
 * @param dst_index Tile index into the destination register.
 * @param operand The input operand logical dataflow buffer id.
 *
 * This function copies a specified number of rows
 * from the srcA or srcB register to the destination register.
 */
template <
    DataCopyType type = DataCopyType::A2D,
    bool IS_32b_DEST_EN = DST_ACCUM_MODE,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(const std::uint32_t dst_index, const std::uint32_t operand) {
    static_assert(
        src_b_bcast_type == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");
    if constexpr (!unpack_to_dest) {
        const std::uint32_t operand_id = get_operand_id(operand);
        const std::uint32_t num_faces = get_operand_num_faces(operand_id);
        const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
        _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
    }
}

/**
 * @brief Performs an eltwise unary datacopy for a block of tiles.
 *
 * @tparam type sets which src register to copy from, values = <A2D, B2D>
 * @tparam IS_32b_DEST_EN set if math destination register is set to Float32/Int32 mode
 * @tparam src_b_bcast_type Broadcast type; carried for API parity with non-Quasar architectures.
 * @tparam unpack_to_dest When true, the unpacker drives dest directly via UNPACR_DEST and the
 *     math thread does not issue the MOV/ELWADD MOP for these tiles. The UNPACK_MATH semaphore
 *     handshake is performed at the compute-API layer (tile_regs_acquire/commit), not here.
 * @param start_dst_index Starting tile index in the destination register.
 * @param ntiles Number of tiles to copy to the destination register.
 * @param operand The input operand logical dataflow buffer id.
 *
 * This function copies a contiguous block of tiles
 * from the srcA or srcB register to the destination register.
 */
template <
    DataCopyType type = DataCopyType::A2D,
    bool IS_32b_DEST_EN = DST_ACCUM_MODE,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_block(
    const std::uint32_t start_dst_index, const std::uint32_t ntiles, const std::uint32_t operand) {
    static_assert(
        src_b_bcast_type == BroadcastType::NONE, "Only BroadcastType::NONE is supported on Quasar right now");
    if constexpr (!unpack_to_dest) {
        const std::uint32_t operand_id = get_operand_id(operand);
        const std::uint32_t num_faces = get_operand_num_faces(operand_id);
        const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);

        for (std::uint32_t dst_index = start_dst_index; dst_index < start_dst_index + ntiles; dst_index++) {
            _llk_math_eltwise_unary_datacopy_(num_faces * face_r_dim, dst_index);
        }
    }
}
