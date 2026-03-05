// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_compressed.h"

// NOTE: Requires custom_mm.h to be included before this header.
// Depends on _llk_math_custom_mm_, semaphore::UNPACK_SYNC, TT_MOP, etc.

namespace compressed {

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
 * @brief Reconfig SrcA format for use inside custom_mm MOP loop.
 *
 * Uses direct cfg[] writes instead of cfg_reg_rmw_tensix to avoid
 * tensix instruction pipeline latency. Must be called AFTER
 * wait_for_next_context() when the unpacker is paused at semaphore.
 */
FORCE_INLINE void reconfig_custom_mm_srca(
    volatile uint* cfg, uint32_t fmt_idx, uint32_t reg0_base, uint32_t reg2_base) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA with pre-resolved DataFormat value (no lookup). */
FORCE_INLINE void reconfig_custom_mm_srca_raw(
    volatile uint* cfg, uint32_t src_format, uint32_t reg0_base, uint32_t reg2_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA input format only (REG0). REG2 stays unchanged. */
FORCE_INLINE void reconfig_custom_mm_srca_input_only(volatile uint* cfg, uint32_t src_format, uint32_t reg0_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
}

// ---------------------------------------------------------------------------
// Constexpr template-unrolled custom MM (zero runtime format lookup overhead)
//
// Format array is passed as positional CTAs. Template recursion unrolls
// the MOP loop so each tile's format is a compile-time constant.
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
 * @brief Unroll all K pairs using fold expression.
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
// Constexpr array, runtime loop (compact code, fast array access)
//
// Format array is passed as constexpr CTAs (same as above), but the loop
// is a regular for-loop instead of template-unrolled. This gives compact
// code size (~1.5KB vs ~10KB) while avoiding slow volatile L1 pointer reads.
// ---------------------------------------------------------------------------

/**
 * @brief Extract format index from packed constexpr array at runtime index.
 */
template <size_t NUM_PACKED>
FORCE_INLINE uint32_t _get_packed_format_(const std::array<uint32_t, NUM_PACKED>& fmt_packed, uint32_t tile_idx) {
    return (fmt_packed[tile_idx / TILES_PER_UINT32] >> ((tile_idx % TILES_PER_UINT32) * ASSIGN_BITS)) & ASSIGN_MASK;
}

/**
 * @brief Custom MM block with constexpr format array but runtime loop.
 *
 * Same interface as custom_mm_compressed_block_constexpr but uses a for-loop
 * instead of template unrolling. Compact code, array lives in local memory.
 */
template <uint32_t KT_DIM, uint32_t CT_DIM, size_t NUM_TILES, const std::array<uint32_t, NUM_TILES>& FMT_FLAT>
FORCE_INLINE void custom_mm_compressed_block_runtime_loop(
    uint32_t addr_in0, uint32_t addr_in1, uint32_t in0_face_r_dim, uint32_t dst_index) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();
        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;
        // REG2 (Out_data_format) stays at initial bfp8 — only change input format (REG0)

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_in0;
        TT_MOP(0, (KT_DIM / 2) - 1, 0);

        uint32_t address_a = addr_in1;
        constexpr uint32_t num_pairs = KT_DIM * CT_DIM / 2;

        // FMT_FLAT is packed: one uint32 per pair
        // [31:24]=size1 [23:16]=size0 [15:8]=fmt1 [7:0]=fmt0
        union PairInfo {
            uint32_t packed;
            struct {
                uint8_t fmt0, fmt1, sz0, sz1;
            };
        };

        for (uint32_t pair = 0; pair < num_pairs; pair++) {
            PairInfo p;
            p.packed = FMT_FLAT[pair];  // single load

            wait_for_next_context(2);
            reconfig_custom_mm_srca_input_only(cfg, p.fmt0, reg0_base);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            address_a += p.sz0;
            semaphore_post(semaphore::UNPACK_SYNC);

            wait_for_next_context(2);
            reconfig_custom_mm_srca_input_only(cfg, p.fmt1, reg0_base);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            address_a += p.sz1;
            semaphore_post(semaphore::UNPACK_SYNC);
        }
    }));
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, KT_DIM, CT_DIM)));
}

// ---------------------------------------------------------------------------
// Runtime version (for fallback / ct_dim>1)
// ---------------------------------------------------------------------------

/**
 * @brief Custom _run_ for compressed weights with per-tile format switching.
 *
 * Supports ct_dim=1 and even ct_dim>1.
 */
FORCE_INLINE void _custom_mm_compressed_run_(
    const volatile uint8_t* assign_ptr, uint32_t address_a, uint32_t address_b, uint32_t kt_dim, uint32_t ct_dim) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();

        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;
        uint32_t reg2_base = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & ~0x0f;

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        TT_MOP(0, (kt_dim / 2) - 1, 0);

        if (ct_dim == 1) {
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
