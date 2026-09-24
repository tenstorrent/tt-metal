// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    ckernel::sfpu::TopkTieOrder TIE_ORDER = ckernel::sfpu::TopkTieOrder::Unset>
inline void calculate_bitonic_topk_phases_steps(
    std::uint32_t idir,
    std::uint32_t i_end_phase,
    std::uint32_t i_start_phase,
    std::uint32_t i_end_step,
    std::uint32_t i_start_step) {
    _bitonic_topk_phases_steps<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED, TIE_ORDER>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool idir = false,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    ckernel::sfpu::TopkTieOrder TIE_ORDER = ckernel::sfpu::TopkTieOrder::Unset,
    std::uint32_t TAG_BITS = 16>
inline void calculate_bitonic_topk_merge(std::uint32_t m_iter, std::uint32_t k) {
    _bitonic_topk_merge<
        APPROXIMATION_MODE,
        is_fp32_dest_acc_en,
        idir,
        STABLE_SORT,
        FUSED,
        RANK_STAMPED,
        TIE_ORDER,
        TAG_BITS>(m_iter, k);
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    ckernel::sfpu::TopkTieOrder TIE_ORDER = ckernel::sfpu::TopkTieOrder::Unset>
inline void calculate_bitonic_topk_rebuild(
    std::uint32_t idir, std::uint32_t m_iter, std::uint32_t k, std::uint32_t logk, std::uint32_t skip_second) {
    _bitonic_topk_rebuild<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED, TIE_ORDER>(
        idir, m_iter, k, logk, skip_second);
}

// Fused-key stable topk sweeps (see _topk_fuse_tile_/_topk_defuse_tile_ in the LLK header).
// largest is the op's GLOBAL sort order; fuse runs once per fresh 2-tile slab, defuse once on the
// final output tiles.
template <bool APPROXIMATION_MODE, bool largest>
inline void calculate_topk_fuse() {
    _topk_fuse_tile_<largest>();
}

template <
    bool APPROXIMATION_MODE,
    bool largest,
    std::uint32_t index_store_mode = static_cast<std::uint32_t>(InstrModLoadStore::INT32)>
inline void calculate_topk_defuse(std::uint32_t num_tiles) {
    _topk_defuse_tile_<largest, index_store_mode>(num_tiles);
}

// Rank-stamped stable topk stamp sweep (see _topk_stamp_local_positions_ in the LLK header).
// The stamp runs once per freshly transposed 2-tile slab, before every local-sort call.
// largest is the op's GLOBAL sort order; TAG_BITS the tag field width (16 for bf16 values).
template <bool APPROXIMATION_MODE, bool largest, std::uint32_t TAG_BITS = 16>
inline void calculate_topk_stamp_local_positions() {
    _topk_stamp_local_positions_<largest, TAG_BITS>();
}

// Single-tile stamp with a runtime rank base: the k>32 insertion cascade's chain-position stamp
// (accumulator tile at level p gets [32p, 32p+32); the fresh chunk gets the top range once, at
// level 0; loser-tile tags ride). See _topk_stamp_tile_rank_range_ in the LLK header.
template <bool APPROXIMATION_MODE, bool largest, std::uint32_t TAG_BITS = 16>
inline void calculate_topk_stamp_tile_rank_range(std::uint32_t dst_tile_index, std::uint32_t rank_base) {
    _topk_stamp_tile_rank_range_<largest, TAG_BITS>(dst_tile_index, rank_base);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_topk_canonicalize_negzero() {
    if constexpr (is_fp32_dest_acc_en && !TOPK_UINT16_IN_FP32_DEST) {
        _topk_canonicalize_negzero_value_tiles_();
    }
}

template <bool APPROXIMATION_MODE, bool FUSED = false, bool RANK_STAMPED = false, std::uint32_t TAG_BITS = 16>
inline void topk_init() {
    static_assert(!(FUSED && RANK_STAMPED), "fused and rank-stamped modes are mutually exclusive");
    static_assert(RANK_STAMPED || TAG_BITS == 16, "TAG_BITS applies to the rank-stamped mode only");
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 32}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (FUSED) {
        _init_topk_fused_();
    } else if constexpr (RANK_STAMPED) {
        _init_topk_rank_stamped_<TAG_BITS>();
    } else {
        _init_topk();
    }
}

// Op class for the bitonic TopK local sort (phases/steps). The kernel walks Dest itself.
// TAG_BITS is used only by init().
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset,
    std::uint32_t TAG_BITS = 16>
struct TopkLocalSort : SfpuUnaryOp<TopkLocalSort<
                           APPROXIMATION_MODE,
                           is_fp32_dest_acc_en,
                           STABLE_SORT,
                           FUSED,
                           RANK_STAMPED,
                           TIE_ORDER,
                           TAG_BITS>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(
        std::uint32_t idir,
        std::uint32_t i_end_phase,
        std::uint32_t i_start_phase,
        std::uint32_t i_end_step,
        std::uint32_t i_start_step) {
        calculate_bitonic_topk_phases_steps<
            APPROXIMATION_MODE,
            is_fp32_dest_acc_en,
            STABLE_SORT,
            FUSED,
            RANK_STAMPED,
            TIE_ORDER>(idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
    }

    static inline __attribute__((always_inline)) void init_op() {
        topk_init<APPROXIMATION_MODE, FUSED, RANK_STAMPED, TAG_BITS>();
    }
};

// Op class for the bitonic TopK merge stage. The kernel walks Dest itself.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool idir = false,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset,
    std::uint32_t TAG_BITS = 16>
struct TopkMerge : SfpuUnaryOp<TopkMerge<
                       APPROXIMATION_MODE,
                       is_fp32_dest_acc_en,
                       idir,
                       STABLE_SORT,
                       FUSED,
                       RANK_STAMPED,
                       TIE_ORDER,
                       TAG_BITS>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(std::uint32_t m_iter, std::uint32_t k) {
        calculate_bitonic_topk_merge<
            APPROXIMATION_MODE,
            is_fp32_dest_acc_en,
            idir,
            STABLE_SORT,
            FUSED,
            RANK_STAMPED,
            TIE_ORDER,
            TAG_BITS>(m_iter, k);
    }
};

// Op class for the bitonic TopK rebuild stage. The kernel walks Dest itself.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset>
struct TopkRebuild
    : SfpuUnaryOp<TopkRebuild<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED, TIE_ORDER>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(
        std::uint32_t idir, std::uint32_t m_iter, std::uint32_t k, std::uint32_t logk, std::uint32_t skip_second) {
        calculate_bitonic_topk_rebuild<
            APPROXIMATION_MODE,
            is_fp32_dest_acc_en,
            STABLE_SORT,
            FUSED,
            RANK_STAMPED,
            TIE_ORDER>(idir, m_iter, k, logk, skip_second);
    }
};

// Op class that fuses a 2-tile TopK slab into packed [bf16 value | u16 index] keys. The kernel walks Dest itself.
template <bool APPROXIMATION_MODE, bool largest>
struct TopkFuse : SfpuUnaryOp<TopkFuse<APPROXIMATION_MODE, largest>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        calculate_topk_fuse<APPROXIMATION_MODE, largest>();
    }
};

// Op class that splits packed TopK keys back into value and index tiles. The kernel walks Dest itself.
template <
    bool APPROXIMATION_MODE,
    bool largest,
    std::uint32_t index_store_mode = static_cast<std::uint32_t>(InstrModLoadStore::INT32)>
struct TopkDefuse : SfpuUnaryOp<TopkDefuse<APPROXIMATION_MODE, largest, index_store_mode>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(std::uint32_t num_tiles) {
        calculate_topk_defuse<APPROXIMATION_MODE, largest, index_store_mode>(num_tiles);
    }
};

// Op class that stamps local rank tags into a 2-tile TopK slab. The kernel walks Dest itself.
template <bool APPROXIMATION_MODE, bool largest, std::uint32_t TAG_BITS = 16>
struct TopkStampLocalPositions : SfpuUnaryOp<TopkStampLocalPositions<APPROXIMATION_MODE, largest, TAG_BITS>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        calculate_topk_stamp_local_positions<APPROXIMATION_MODE, largest, TAG_BITS>();
    }
};

// Op class that stamps one TopK value tile with rank tags from a caller-chosen base. The kernel walks Dest itself.
template <bool APPROXIMATION_MODE, bool largest, std::uint32_t TAG_BITS = 16>
struct TopkStampTileRankRange : SfpuUnaryOp<TopkStampTileRankRange<APPROXIMATION_MODE, largest, TAG_BITS>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate(std::uint32_t dst_tile_index, std::uint32_t rank_base) {
        calculate_topk_stamp_tile_rank_range<APPROXIMATION_MODE, largest, TAG_BITS>(dst_tile_index, rank_base);
    }
};

// Op class that folds -0.0 into +0.0 in the two value tiles of a TopK slab. The kernel walks Dest itself.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
struct TopkCanonicalizeNegzero : SfpuUnaryOp<TopkCanonicalizeNegzero<APPROXIMATION_MODE, is_fp32_dest_acc_en>> {
    static constexpr bool walks_faces = false;

    static inline __attribute__((always_inline)) void calculate() {
        calculate_topk_canonicalize_negzero<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
    }
};

}  // namespace sfpu
}  // namespace ckernel
