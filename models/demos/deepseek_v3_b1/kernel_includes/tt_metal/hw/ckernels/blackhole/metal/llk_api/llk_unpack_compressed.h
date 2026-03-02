// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOTE: This header is only for TRISC (compute) kernels.
// It must be included AFTER the standard compute API headers
// (eltwise_binary.h, tile_move_copy.h, etc.) to avoid symbol conflicts.
// It relies on _llk_unpack_AB_, _llk_math_eltwise_binary_, etc. being already declared.

/*************************************************************************
 * LLK UNPACK FOR COMPRESSED TENSORS
 *
 * Compressed tensors have per-tile data formats (bfp8/bfp4/bfp2/bfp0).
 * The assignment array (2 bits per tile) tells each tile's format.
 *
 * These APIs bypass the constexpr arrays (unpack_src_format, unpack_tile_size,
 * fifo_page_size, etc.) and instead use runtime format/size from lookup tables.
 * This allows per-tile format switching with a single CB.
 *
 * The "in1_compressed" naming convention means operand B (in1) is the
 * compressed tensor, while operand A (in0) is a normal tiled tensor.
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
constexpr uint32_t TILES_PER_BYTE = 8 / ASSIGN_BITS;  // 4

// Format index for bfp0 (zero tile)
constexpr uint32_t FMT_BFP0 = 3;

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
 *
 * Sets HW registers for src_format, dst_format, tile_size, face dims.
 * dst_format = src_format for BFP formats (HW handles BFP→float internally).
 * Uses DST_ACCUM_MODE (compile-time constant from kernel config).
 */
FORCE_INLINE void reconfig_unpack_srca(uint32_t fmt_idx) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    UNPACK((_llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE>(
        src_format, src_format, TILE_SIZES[fmt_idx], FACE_R_DIM, 4)));
}

/**
 * @brief Reconfigure unpacker B for a specific compressed format.
 */
FORCE_INLINE void reconfig_unpack_srcb(uint32_t fmt_idx) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE>(
        src_format, src_format, TILE_SIZES[fmt_idx], FACE_R_DIM, 4)));
}

// ---------------------------------------------------------------------------
// Init + Add tiles with in1 compressed
//
// These replace add_tiles_init / add_tiles for the case where in1 (srcB)
// is a compressed tensor. They bypass constexpr array reads for in1 and
// use explicit addresses / runtime format params instead.
// ---------------------------------------------------------------------------

/**
 * @brief Initialize for eltwise add where in1 is compressed.
 *
 * Unlike add_tiles_init which reads tile shape, data format, and page size
 * from constexpr arrays for both CBs, this only reads from cb_in0 (normal tensor).
 * For in1 (compressed), it uses the standard 32x32 tile shape directly.
 *
 * @param cb_in0 CB index for operand A (normal bf16 tiled tensor)
 */
FORCE_INLINE void add_tiles_init_in1_compressed(uint32_t cb_in0) {
    UNPACK(({
        const ckernel::TensorShape ts = get_operand_tensor_shape(get_operand_id(cb_in0));
        _llk_unpack_AB_init_<BroadcastType::NONE>(ts, 0 /*transpose*/);
    }));
    MATH(({
        const ckernel::TensorShape ts = get_operand_tensor_shape(get_operand_id(cb_in0));
        _llk_math_eltwise_binary_init_<ELWADD, NONE, MATH_FIDELITY>(ts, 0 /*acc_to_dest*/);
    }));
}

/**
 * @brief Add two tiles at explicit L1 addresses.
 *
 * Replaces add_tiles() when in1 is compressed. Both addresses are provided
 * explicitly — no CB page_size lookups.
 *
 * The caller must:
 *   1. Call reconfig_unpack_srca() before this for the correct in1 format
 *   2. Compute both L1 addresses (in >> cb_addr_shift units)
 *
 * @param addr_a L1 address of tile A (in >> cb_addr_shift units)
 * @param addr_b L1 address of compressed tile B (in >> cb_addr_shift units)
 * @param dst_index Destination register index
 */
FORCE_INLINE void add_tiles_in1_compressed(uint32_t cb_in0, uint32_t addr_a, uint32_t addr_b, uint32_t dst_index) {
    UNPACK((_llk_unpack_AB_<BroadcastType::NONE>(addr_a, addr_b)));
    MATH(({
        const ckernel::TensorShape ts = get_operand_tensor_shape(get_operand_id(cb_in0));
        _llk_math_eltwise_binary_<
            ELWADD,
            NONE,
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            MATH_FIDELITY,
            EltwiseBinaryReuseDestType::NONE>(ts, dst_index, true);
    }));
}

}  // namespace compressed
