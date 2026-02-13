// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"
#include "llk_assert.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t transpose = 0) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    // If there is no broadcast, operands must be the same shape. With broadcast, operand B may have different tile
    // dims.
    if constexpr (BType == BroadcastType::NONE) {
        LLK_ASSERT(
            get_operand_num_faces(operandA_id) == get_operand_num_faces(operandB_id),
            "Operands must have same num_faces when BType == NONE");
        LLK_ASSERT(
            get_operand_face_r_dim(operandA_id) == get_operand_face_r_dim(operandB_id),
            "Operands must have same face_r_dim when BType == NONE");
    }
    LLK_ASSERT(
        get_operand_src_format(operandA_id) == get_operand_src_format(operandB_id),
        "Operands must have same src format");
    LLK_ASSERT(
        get_operand_dst_format(operandA_id) == get_operand_dst_format(operandB_id),
        "Operands must have same dst format");

    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);  // face r dim in unpA and unpB are the same
    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const bool narrow_tile =
        get_operand_narrow_tile(operandA_id);  // if narrow tile read face 0 twice for row broadcast

    _llk_unpack_AB_init_<BType>(face_r_dim, num_faces, narrow_tile, transpose);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t bcast_row_idx = 0) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_a = get_local_cb_interface(operandA_id).fifo_page_size * tile_index_a;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_b = get_local_cb_interface(operandB_id).fifo_page_size * tile_index_b;
    std::uint32_t address_b = base_address_b + offset_address_b;

    // For row broadcast with non-zero row index, adjust address to point to the desired row
    if constexpr (BType == BroadcastType::ROW) {
        if (bcast_row_idx > 0) {
            // Row broadcast reads a full 32-element row, which spans two faces:
            //   Row 0: Face0 row 0 (cols 0-15) + Face1 row 0 (cols 16-31)
            //   Row 31: Face2 row 15 (cols 0-15) + Face3 row 15 (cols 16-31)
            //
            // Within each face, rows are stored contiguously

            // Get the data format to calculate bytes per element
            const std::uint32_t src_format = unpack_src_format[operandB_id];
            // NOTE: This non-zero row-index row-broadcast addressing path has only been validated for BF16 (Float16_b).
            LLK_ASSERT(
                src_format == static_cast<std::uint32_t>(DataFormat::Float16_b),
                "Non-zero row-index row broadcast only supported for DataFormat::Float16_b (BF16)");

            // Use existing FACE_WIDTH and FACE_HEIGHT constants from ckernel_defs.h
            const std::uint32_t bytes_per_row_in_face = SCALE_DATUM_SIZE(src_format, FACE_WIDTH);
            const std::uint32_t bytes_per_face = SCALE_DATUM_SIZE(src_format, FACE_WIDTH * FACE_HEIGHT);

            std::uint32_t row_offset_bytes;
            if (bcast_row_idx < FACE_HEIGHT) {
                // Rows 0-15 are in Face 0/1
                // Offset to the row within Face 0
                row_offset_bytes = bcast_row_idx * bytes_per_row_in_face;
            } else {
                // Rows 16-31 are in Face 2/3
                // Skip first two faces, then offset to the row within Face 2
                row_offset_bytes = 2 * bytes_per_face + (bcast_row_idx - FACE_HEIGHT) * bytes_per_row_in_face;
            }

            // Convert to 16B units for L1 address alignment
            std::uint32_t row_offset_16B_units = row_offset_bytes >> 4;

            address_b += row_offset_16B_units;
        }
    }

    WAYPOINT("UABW");
    _llk_unpack_AB_<BType>(address_a, address_b);
    WAYPOINT("UABD");
}

template <ReduceDim dim, BroadcastType BType = BroadcastType::NONE, bool enforce_fp32_accumulation = false>
inline void llk_unpack_AB_reduce_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t within_face_16x16_transpose = 0) {
    LLK_ASSERT(transpose == within_face_16x16_transpose, "transpose and within_face_16x16_transpose must match");
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    // If there is no broadcast, operands must be the same shape. With broadcast, operand B may have different tile
    // dims.
    if constexpr (BType == BroadcastType::NONE) {
        LLK_ASSERT(
            get_operand_num_faces(operandA_id) == get_operand_num_faces(operandB_id),
            "Operands must have same num_faces when BType == NONE");
        LLK_ASSERT(
            get_operand_face_r_dim(operandA_id) == get_operand_face_r_dim(operandB_id),
            "Operands must have same face_r_dim when BType == NONE");
    }
    LLK_ASSERT(
        get_operand_src_format(operandA_id) == get_operand_src_format(operandB_id),
        "Operands must have same src format");
    LLK_ASSERT(
        get_operand_dst_format(operandA_id) == get_operand_dst_format(operandB_id),
        "Operands must have same dst format");

    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);  // face r dim in unpA and unpB are the same
    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const bool narrow_tile =
        get_operand_narrow_tile(operandA_id);  // if narrow tile read face 0 twice for row broadcast
    // TODO NC: Move to TRISC1 tt-metal#36411
    if constexpr (enforce_fp32_accumulation) {
        // Set necessary config regs for MOVB2D hi16/lo16 to work
        _llk_unpack_dbg_feature_disable_();
    }
    _llk_unpack_AB_reduce_init_<dim, BType, enforce_fp32_accumulation>(
        face_r_dim, num_faces, narrow_tile, transpose, within_face_16x16_transpose);
}
