// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK FOR COMPRESSED TENSORS
 *
 * Compressed tensors have per-tile data formats (bfp8/bfp4/bfp2/bfp0).
 * The assignment array (2 bits per tile) tells each tile's format.
 * These APIs allow runtime reconfiguration of the unpacker per tile,
 * without requiring separate CBs per format.
 *
 * Format index mapping (matches COMPRESSED_FORMATS):
 *   0 = bfp8 (DataFormat::Bfp8_b)
 *   1 = bfp4 (DataFormat::Bfp4_b)
 *   2 = bfp2 (DataFormat::Bfp2_b)
 *   3 = bfp0 (zero tile, no data)
 *************************************************************************/

namespace compressed {

// Tile sizes in bytes for each format (32x32 tile, 4 faces of 16x16)
constexpr uint32_t TILE_SIZES[] = {
    1088,  // bfp8: 64 exp + 1024 data
    576,   // bfp4: 64 exp + 512 data
    320,   // bfp2: 64 exp + 256 data
    0,     // bfp0: no data
};

// DataFormat values for each compressed format index
constexpr uint32_t DATA_FORMATS[] = {
    static_cast<uint32_t>(DataFormat::Bfp8_b),
    static_cast<uint32_t>(DataFormat::Bfp4_b),
    static_cast<uint32_t>(DataFormat::Bfp2_b),
    0,  // bfp0: unused
};

// Bits per assignment entry
constexpr uint32_t ASSIGN_BITS = 2;
constexpr uint32_t ASSIGN_MASK = (1 << ASSIGN_BITS) - 1;
constexpr uint32_t TILES_PER_BYTE = 8 / ASSIGN_BITS;  // 4

/**
 * @brief Read a tile's format index from the packed assignment array.
 *
 * Assignment is 2-bit packed: 4 tiles per byte, LSB first.
 *
 * @param assignment_ptr Pointer to the packed assignment array in L1.
 * @param tile_id Linear tile index (row-major).
 * @return Format index (0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0).
 */
FORCE_INLINE uint32_t get_tile_format(const volatile uint8_t* assignment_ptr, uint32_t tile_id) {
    uint32_t byte_idx = tile_id / TILES_PER_BYTE;
    uint32_t bit_offset = (tile_id % TILES_PER_BYTE) * ASSIGN_BITS;
    return (assignment_ptr[byte_idx] >> bit_offset) & ASSIGN_MASK;
}

/**
 * @brief Get the packed tile size in bytes for a given format index.
 */
FORCE_INLINE uint32_t get_compressed_tile_size(uint32_t fmt_idx) { return TILE_SIZES[fmt_idx]; }

/**
 * @brief Reconfigure unpacker A for a specific compressed format.
 *
 * Calls the existing _llk_unpack_reconfig_data_format_srca_impl_ with
 * the format-specific parameters. Does nothing for bfp0 (zero tiles).
 *
 * The dst_format is set to the same BFP format as src — the HW handles
 * BFP→float conversion internally. This matches how get_single_unpack_dst_format
 * works for non-Float32 source formats (dst = src).
 *
 * @param fmt_idx Format index from get_tile_format().
 */
template <bool is_fp32_dest_acc_en>
FORCE_INLINE void reconfig_unpack_srca(uint32_t fmt_idx) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en>(
        src_format,
        src_format,  // dst_format = src_format for BFP formats
        TILE_SIZES[fmt_idx],
        FACE_R_DIM,  // 16 for standard tiles
        4            // 4 faces for 32x32 tile
    );
}

/**
 * @brief Reconfigure unpacker B for a specific compressed format.
 *
 * @param fmt_idx Format index from get_tile_format().
 */
template <bool is_fp32_dest_acc_en>
FORCE_INLINE void reconfig_unpack_srcb(uint32_t fmt_idx) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en>(
        src_format,
        src_format,  // dst_format = src_format for BFP formats
        TILE_SIZES[fmt_idx],
        FACE_R_DIM,
        4);
}

}  // namespace compressed
