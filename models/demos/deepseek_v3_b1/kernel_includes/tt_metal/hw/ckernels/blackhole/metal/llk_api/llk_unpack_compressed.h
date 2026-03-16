// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOTE: This header is only for TRISC (compute) kernels.
// It must be included AFTER the standard compute API headers
// to avoid symbol conflicts.

/*************************************************************************
 * LLK UNPACK FOR COMPRESSED TENSORS — shared definitions
 *
 * Compressed tensors have per-tile data formats (bfp8/bfp4/bfp2/bfp0).
 * The assignment array (2 bits per tile) tells each tile's format.
 *
 * Format index mapping (matches COMPRESSED_FORMATS):
 *   0 = bfp8 (DataFormat::Bfp8_b)
 *   1 = bfp4 (DataFormat::Bfp4_b)
 *   2 = bfp2 (DataFormat::Bfp2_b)
 *   3 = bfp0 (zero tile, no data — caller must skip)
 *************************************************************************/

namespace compressed {

// Tile sizes in bytes for each format (32x32 tile, 4 faces of 16x16)
constexpr uint32_t TILE_SIZES[] = {
    1088,  // bfp8: 64 exp + 1024 data
    576,   // bfp4: 64 exp + 512 data
    320,   // bfp2: 64 exp + 256 data
    0,     // bfp0: no data
};

// Tile sizes pre-shifted for CB address arithmetic (>> cb_addr_shift, i.e. >> 4)
constexpr uint32_t TILE_SIZES_SHIFTED[] = {
    1088 >> 4,  // bfp8: 68
    576 >> 4,   // bfp4: 36
    320 >> 4,   // bfp2: 20
    0,          // bfp0
};

// DataFormat values for each compressed format index
constexpr uint32_t DATA_FORMATS[] = {
    static_cast<uint32_t>(DataFormat::Bfp8_b),
    static_cast<uint32_t>(DataFormat::Bfp4_b),
    static_cast<uint32_t>(DataFormat::Bfp2_b),
    0,  // bfp0: unused
};

// Assignment packing constants
constexpr uint32_t ASSIGN_BITS = 2;
constexpr uint32_t ASSIGN_MASK = (1 << ASSIGN_BITS) - 1;
constexpr uint32_t TILES_PER_BYTE = 8 / ASSIGN_BITS;     // 4
constexpr uint32_t TILES_PER_UINT32 = 32 / ASSIGN_BITS;  // 16

// Format index for bfp0 (zero tile)
constexpr uint32_t FMT_BFP0 = 3;

// Address of 512-byte all-zeros region in L1, pre-shifted for THCON registers (>> cb_addr_shift=4).
// Zero tiles point here instead of the weight data buffer.
constexpr uint32_t ZEROS_ADDR_SHIFTED = MEM_ZEROS_BASE >> 4;

// ---------------------------------------------------------------------------
// Assignment helpers
// ---------------------------------------------------------------------------

/**
 * @brief Read a tile's format index from the packed assignment array.
 *
 * Assignment is 2-bit packed: 4 tiles per byte, LSB first.
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

// ---------------------------------------------------------------------------
// Unpacker reconfig (per-tile format switching)
// ---------------------------------------------------------------------------

/**
 * @brief Reconfigure unpacker A for a specific compressed format.
 */
FORCE_INLINE void reconfig_unpack_srca(uint32_t fmt_idx) {
    UNPACK(({
        uint32_t src_format = DATA_FORMATS[fmt_idx];
        uint32_t tile_size_shifted = TILE_SIZES[fmt_idx] >> 4;
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE>(
            src_format, src_format, tile_size_shifted, FACE_R_DIM, 4);
    }));
}

/**
 * @brief Reconfigure unpacker B for a specific compressed format.
 */
FORCE_INLINE void reconfig_unpack_srcb(uint32_t fmt_idx) {
    UNPACK(({
        uint32_t src_format = DATA_FORMATS[fmt_idx];
        uint32_t tile_size_shifted = TILE_SIZES[fmt_idx] >> 4;
        _llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE>(
            src_format, src_format, tile_size_shifted, FACE_R_DIM, 4);
    }));
}

}  // namespace compressed
