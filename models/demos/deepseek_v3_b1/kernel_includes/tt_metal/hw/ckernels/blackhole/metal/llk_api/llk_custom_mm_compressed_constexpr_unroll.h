// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "llk_custom_mm_compressed_common.h"

namespace compressed {

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

}  // namespace compressed
