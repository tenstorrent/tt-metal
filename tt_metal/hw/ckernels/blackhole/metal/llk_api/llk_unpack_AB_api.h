// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_mop_config(const bool transpose_of_faces = false, const std::uint32_t operand_id = 0) {
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);  // if narrow tile read face 0 twice for row broadcast
                                                                   // or read face 0 and 1 for col broadcast
    _llk_unpack_AB_mop_config_<BType>(transpose_of_faces, num_faces, narrow_tile);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t transpose = 0) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operandA_id);  // face r dim in unpA and unpB are the same
    const std::uint32_t num_faces = get_operand_num_faces(operandA_id);
    const bool narrow_tile =
        get_operand_narrow_tile(operandA_id);  // if narrow tile read face 0 twice for row broadcast

    LLK_ASSERT(
        (are_unpacker_AB_configured_correctly(
            unpack_src_format[operandA_id],
            unpack_dst_format[operandA_id],
            unpack_src_format[get_operand_id(operandB)],
            unpack_dst_format[get_operand_id(operandB)],
            face_r_dim,
            get_operand_face_r_dim(get_operand_id(operandB)),
            num_faces,
            get_operand_num_faces(get_operand_id(operandB)))),
        "");

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

    LLK_ASSERT(
        (are_unpacker_AB_configured_correctly(
            unpack_src_format[operandA_id],
            unpack_dst_format[operandA_id],
            unpack_src_format[operandB_id],
            unpack_dst_format[operandB_id],
            get_operand_face_r_dim(operandA_id),
            get_operand_face_r_dim(operandB_id),
            get_operand_num_faces(operandA_id),
            get_operand_num_faces(operandB_id))),
        "");

    // For row broadcast with non-zero row index, adjust address to point to the desired row
    if constexpr (BType == BroadcastType::ROW) {
        if (bcast_row_idx > 0) {
            // Row broadcast reads a full 32-element row, which spans two faces:
            //   Row 0: Face0 row 0 (cols 0-15) + Face1 row 0 (cols 16-31)
            //   Row 31: Face2 row 15 (cols 0-15) + Face3 row 15 (cols 16-31)
            //
            // Within each face, rows are stored contiguously

            // Get the data format to calculate bytes per element
            const uint32_t src_format = unpack_src_format[operandB_id];

            // Use existing FACE_WIDTH and FACE_HEIGHT constants from ckernel_defs.h
            const uint32_t bytes_per_row_in_face = SCALE_DATUM_SIZE(src_format, FACE_WIDTH);
            const uint32_t bytes_per_face = SCALE_DATUM_SIZE(src_format, FACE_WIDTH * FACE_HEIGHT);

            uint32_t row_offset_bytes;
            if (bcast_row_idx < FACE_HEIGHT) {
                // Rows 0-15 are in Face 0/1
                // Offset to the row within Face 0
                row_offset_bytes = bcast_row_idx * bytes_per_row_in_face;
            } else {
                // Rows 16-31 are in Face 2/3
                // Skip first two faces, then offset to the row within Face 2
                row_offset_bytes = 2 * bytes_per_face + (bcast_row_idx - FACE_HEIGHT) * bytes_per_row_in_face;
            }

            uint32_t row_offset_16B_units = row_offset_bytes >> 4;

            address_b += row_offset_16B_units;
        }
    }

    WAYPOINT("UABW");
    _llk_unpack_AB_<BType>(address_a, address_b);
    WAYPOINT("UABD");
}
