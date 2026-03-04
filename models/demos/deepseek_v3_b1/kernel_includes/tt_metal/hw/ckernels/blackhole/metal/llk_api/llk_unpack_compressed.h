// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <utility>

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
constexpr uint32_t TILES_PER_BYTE = 8 / ASSIGN_BITS;     // 4
constexpr uint32_t TILES_PER_UINT32 = 32 / ASSIGN_BITS;  // 16

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

// ---------------------------------------------------------------------------
// Matmul with compressed in1 (weights)
//
// In matmul: in0 (activations) → srcB, in1 (weights) → srcA (swapped!)
// So compressed weights need reconfig_unpack_srca, not srcb.
// These APIs use explicit addresses for the compressed weight tiles.
// ---------------------------------------------------------------------------

/**
 * @brief Matmul one A tile with one compressed B tile at explicit addresses.
 *
 * Calls _llk_unpack_AB_matmul_ with explicit base addresses and tile sizes.
 * The caller must:
 *   1. Call reconfig_unpack_srca() before this for the correct B format
 *   2. Call mm_init_short() once before the matmul loop
 *
 * @param addr_a Base address of A (in0, activations, srcB) in shifted units
 * @param addr_b Address of compressed B tile (in1, weights, srcA) in shifted units
 * @param tile_index_a Tile index within A (along K dimension)
 * @param tile_size_a Page size of A tiles in shifted units
 * @param tile_size_b Size of compressed B tile in shifted units
 * @param partial_face_a Whether in1 (srcA/weights) has partial face
 * @param partial_face_b Whether in0 (srcB/activations) has partial face
 * @param dst_index Destination register index (output column)
 */
/**
 * @brief Matmul one A tile with one compressed B tile.
 *
 * Reads the format from the assignment array, skips bfp0 tiles,
 * reconfigures srcA unpacker, then does unpack + math.
 * Returns the compressed tile size in shifted units (0 for bfp0).
 *
 * @param assign_ptr Pointer to packed 2-bit assignment array
 * @param tile_idx Linear tile index in row-major order
 * @param addr_a Base address of A (in0, activations, srcB) in shifted units
 * @param addr_b Address of compressed B tile (in1, weights, srcA) in shifted units
 * @param tile_index_a Tile index within A (along K dimension)
 * @param tile_size_a Page size of A tiles in shifted units
 * @param partial_face_a Whether in1 (srcA/weights) has partial face
 * @param partial_face_b Whether in0 (srcB/activations) has partial face
 * @param dst_index Destination register index (output column)
 * @return Tile size in shifted units (for advancing addr_b)
 */
FORCE_INLINE uint32_t matmul_tiles_in1_compressed(
    const volatile uint8_t* assign_ptr,
    uint32_t tile_idx,
    uint32_t addr_a,
    uint32_t addr_b,
    uint32_t tile_index_a,
    uint32_t tile_size_a,
    bool partial_face_a,
    bool partial_face_b,
    uint32_t dst_index) {
    uint32_t fmt = get_tile_format(assign_ptr, tile_idx);
    uint32_t tile_size_b = TILE_SIZES[fmt] >> cb_addr_shift;

    if (fmt != FMT_BFP0) {
        reconfig_unpack_srca(fmt);
        UNPACK((_llk_unpack_AB_matmul_(
            addr_a,
            addr_b,
            tile_index_a,
            0,  // tile_index_b: addr_b already points to the tile
            tile_size_a,
            tile_size_b,
            partial_face_a,
            partial_face_b)));
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(dst_index)));
    }

    return tile_size_b;
}

// ---------------------------------------------------------------------------
// Custom MM with compressed in1 (per-tile format reconfig in software loop)
//
// Uses custom_mm_block init/uninit for MOP setup, but replaces the standard
// _run_ with a custom version that does reconfig_unpack_srca + variable
// address increment per context. ct_dim=1 only. kt_dim must be even.
//
// SrcB (in0/activations) auto-advances via HW counters.
// SrcA (in1/weights) address and format set per context in software loop.
// ---------------------------------------------------------------------------

/**
 * @brief Custom _run_ for ct_dim=1 with per-tile format switching.
 *
 * Replaces _llk_unpack_AB_custom_mm_run_ for compressed weights.
 * Each K-tile can have a different BFP format.
 */
/**
 * @brief Reconfig SrcA format for use inside custom_mm MOP loop.
 *
 * Uses direct cfg[] writes instead of cfg_reg_rmw_tensix to avoid
 * tensix instruction pipeline latency. Must be called AFTER
 * wait_for_next_context() when the unpacker is paused at semaphore.
 */
/**
 * @brief Reconfig SrcA format for use inside custom_mm MOP loop.
 *
 * Uses direct cfg[] writes instead of cfg_reg_rmw_tensix to avoid
 * tensix instruction pipeline latency. Must be called AFTER
 * wait_for_next_context() when the unpacker is paused at semaphore.
 *
 * @param cfg Config register pointer
 * @param fmt_idx Format index (0=bfp8, 1=bfp4, 2=bfp2)
 * @param reg0_base Cached upper bits of THCON_SEC0_REG0 (format bits cleared)
 * @param reg2_base Cached upper bits of THCON_SEC0_REG2 (format bits cleared)
 */
FORCE_INLINE void reconfig_custom_mm_srca(
    volatile uint* cfg, uint32_t fmt_idx, uint32_t reg0_base, uint32_t reg2_base) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
    // NOTE: TT_SETDMAREG for TILE_SIZE_A not needed for custom_mm —
    // custom_mm uses explicit tile_size passed via _run_, not the GPR.
    // The GPR is only used by standard matmul's _llk_unpack_AB_matmul_.
}

// ---------------------------------------------------------------------------
// Constexpr template-unrolled custom MM (zero runtime format lookup overhead)
//
// Format array is passed as positional CTAs. Template recursion unrolls
// the MOP loop so each tile's format is a compile-time constant.
// CTA_BASE is the positional CTA index where format values start.
// ---------------------------------------------------------------------------

/**
 * @brief Process one pair of K tiles (ctx0 + ctx1) with compile-time format.
 */
template <size_t PAIR_IDX, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void _custom_mm_compressed_pair_(
    volatile uint* cfg, uint32_t& address_a, uint32_t reg0_base, uint32_t reg2_base) {
    UNPACK(({
        constexpr uint32_t K = PAIR_IDX * 2;
        constexpr uint32_t fmt0 =
            (FMT_PACKED[K / TILES_PER_UINT32] >> ((K % TILES_PER_UINT32) * ASSIGN_BITS)) & ASSIGN_MASK;
        constexpr uint32_t sz0 = TILE_SIZES[fmt0] >> cb_addr_shift;
        constexpr uint32_t fmt1 =
            (FMT_PACKED[(K + 1) / TILES_PER_UINT32] >> (((K + 1) % TILES_PER_UINT32) * ASSIGN_BITS)) & ASSIGN_MASK;
        constexpr uint32_t sz1 = TILE_SIZES[fmt1] >> cb_addr_shift;

        wait_for_next_context(2);
        reconfig_custom_mm_srca(cfg, fmt0, reg0_base, reg2_base);
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        address_a += sz0;
        semaphore_post(semaphore::UNPACK_SYNC);

        wait_for_next_context(2);
        reconfig_custom_mm_srca(cfg, fmt1, reg0_base, reg2_base);
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        address_a += sz1;
        semaphore_post(semaphore::UNPACK_SYNC);
    }));
}

/**
 * @brief Unroll all K pairs using fold expression (no recursion).
 */
template <size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED, size_t... PAIR_IDXS>
FORCE_INLINE void _custom_mm_compressed_unroll_impl_(
    volatile uint* cfg,
    uint32_t& address_a,
    uint32_t reg0_base,
    uint32_t reg2_base,
    std::index_sequence<PAIR_IDXS...>) {
    (_custom_mm_compressed_pair_<PAIR_IDXS, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base), ...);
}

/**
 * @brief Unroll all context pairs for KT_DIM × CT_DIM tiles.
 *
 * Total pairs = KT_DIM * CT_DIM / 2. Tiles are row-major in FMT_PACKED.
 */
template <uint32_t KT_DIM, uint32_t CT_DIM, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void _custom_mm_compressed_unroll_(
    volatile uint* cfg, uint32_t& address_a, uint32_t reg0_base, uint32_t reg2_base) {
    _custom_mm_compressed_unroll_impl_<NUM_PACKED, FMT_PACKED>(
        cfg, address_a, reg0_base, reg2_base, std::make_index_sequence<KT_DIM * CT_DIM / 2>{});
}

/**
 * @brief Constexpr custom MM block. Zero runtime format lookup.
 *
 * FMT_PACKED is a constexpr array of packed format words (from fill_cta_array).
 * Tiles packed row-major: (k=0,n=0), (k=0,n=1), ..., (k=1,n=0), ...
 * Total tiles = KT_DIM * CT_DIM, must be even.
 */
template <uint32_t KT_DIM, uint32_t CT_DIM, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void custom_mm_compressed_block_constexpr(
    uint32_t addr_in0, uint32_t addr_in1, uint32_t in0_face_r_dim, uint32_t dst_index) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();
        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;
        uint32_t reg2_base = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & ~0x0f;

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_in0;
        TT_MOP(0, (KT_DIM / 2) - 1, 0);

        uint32_t address_a = addr_in1;
        _custom_mm_compressed_unroll_<KT_DIM, CT_DIM, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base);
    }));
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, KT_DIM, CT_DIM)));
}

// ---------------------------------------------------------------------------
// Runtime version (original, for fallback / ct_dim>1)
// ---------------------------------------------------------------------------

/**
 * @brief Custom _run_ for compressed weights with per-tile format switching.
 *
 * Supports ct_dim=1 and even ct_dim>1. Follows the same structure as
 * _llk_unpack_AB_custom_mm_run_ but with per-tile reconfig + variable increment.
 *
 * B[K, N] tiles are row-major in assign_ptr: tile (k, n) at index k * ct_dim + n.
 * Compressed data is also row-major contiguous.
 */
FORCE_INLINE void _custom_mm_compressed_run_(
    const volatile uint8_t* assign_ptr,
    uint32_t address_a,  // starting SrcA address (in1/weights)
    uint32_t address_b,  // SrcB base (in0/activations, set once)
    uint32_t kt_dim,     // must be even
    uint32_t ct_dim) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();

        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;
        uint32_t reg2_base = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & ~0x0f;

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        TT_MOP(0, (kt_dim / 2) - 1, 0);

        if (ct_dim == 1) {
            // ct_dim=1: 2 K-tiles per MOP iteration (ctx0 + ctx1)
            for (uint32_t k = 0; k < kt_dim; k += 2) {
                uint32_t fmt0 = get_tile_format(assign_ptr, k);
                wait_for_next_context(2);
                reconfig_custom_mm_srca(cfg, fmt0, reg0_base, reg2_base);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
                address_a += TILE_SIZES[fmt0] >> cb_addr_shift;
                semaphore_post(semaphore::UNPACK_SYNC);

                uint32_t fmt1 = get_tile_format(assign_ptr, k + 1);
                wait_for_next_context(2);
                reconfig_custom_mm_srca(cfg, fmt1, reg0_base, reg2_base);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
                address_a += TILE_SIZES[fmt1] >> cb_addr_shift;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
        } else {
            // even ct_dim>1: 1 K-step per MOP iteration, ct_dim columns
            // Tiles are row-major: (k=0,n=0), (k=0,n=1), ..., (k=1,n=0), ...
            uint32_t tile_idx = 0;
            for (uint32_t k = 0; k < kt_dim; k++) {
                uint32_t row_start = address_a;
                for (uint32_t ct = 0; ct < ct_dim; ct += 2) {
                    uint32_t fmt0 = get_tile_format(assign_ptr, tile_idx);
                    wait_for_next_context(2);
                    reconfig_custom_mm_srca(cfg, fmt0, reg0_base, reg2_base);
                    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
                    address_a += TILE_SIZES[fmt0] >> cb_addr_shift;
                    tile_idx++;
                    semaphore_post(semaphore::UNPACK_SYNC);

                    uint32_t fmt1 = get_tile_format(assign_ptr, tile_idx);
                    wait_for_next_context(2);
                    reconfig_custom_mm_srca(cfg, fmt1, reg0_base, reg2_base);
                    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
                    address_a += TILE_SIZES[fmt1] >> cb_addr_shift;
                    tile_idx++;
                    semaphore_post(semaphore::UNPACK_SYNC);
                }
            }
        }
    }));
}

/**
 * @brief Custom MM block for compressed weights.
 *
 * Supports ct_dim=1 and even ct_dim>1.
 */
FORCE_INLINE void custom_mm_compressed_block(
    const volatile uint8_t* assign_ptr,
    uint32_t addr_in0,
    uint32_t addr_in1,
    uint32_t in0_face_r_dim,
    uint32_t kt_dim,
    uint32_t ct_dim,
    uint32_t dst_index) {
    UNPACK(({
        wait_for_next_context(1);
        reset_config_context();
    }));
    _custom_mm_compressed_run_(assign_ptr, addr_in1, addr_in0, kt_dim, ct_dim);
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, kt_dim, ct_dim)));
}

}  // namespace compressed
