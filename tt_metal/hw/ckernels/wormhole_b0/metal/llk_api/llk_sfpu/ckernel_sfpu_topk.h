// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_sfpu_op.h"

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

// TopkLocalSort / TopkMerge / TopkRebuild <APPROX, ..., DST_SYNC, DST_ACCUM, FUSED, RANK_STAMPED, TIE_ORDER,
// TAG_BITS>: topk_local_sort, topk_merge, topk_rebuild and topk_tile_init (api/compute/topk.h). All three stages
// share topk_init<APPROX, FUSED, RANK_STAMPED, TAG_BITS>; TAG_BITS only reaches the merge kernel.
template <
    bool APPROXIMATION_MODE,
    bool STABLE_SORT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset,
    std::uint32_t TAG_BITS = 16>
struct TopkLocalSort
    : SfpuUnaryOp<
          TopkLocalSort<APPROXIMATION_MODE, STABLE_SORT, DST_SYNC, DST_ACCUM, FUSED, RANK_STAMPED, TIE_ORDER, TAG_BITS>,
          DST_SYNC,
          DST_ACCUM> {
    static void kernel(
        uint32_t idir, uint32_t i_end_phase, uint32_t i_start_phase, uint32_t i_end_step, uint32_t i_start_step) {
        calculate_bitonic_topk_phases_steps<APPROXIMATION_MODE, DST_ACCUM, STABLE_SORT, FUSED, RANK_STAMPED, TIE_ORDER>(
            idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
    }

    static void init_kernel() { topk_init<APPROXIMATION_MODE, FUSED, RANK_STAMPED, TAG_BITS>(); }
};

template <
    bool APPROXIMATION_MODE,
    bool IDIR,
    bool STABLE_SORT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset,
    std::uint32_t TAG_BITS = 16>
struct TopkMerge : SfpuUnaryOp<
                       TopkMerge<
                           APPROXIMATION_MODE,
                           IDIR,
                           STABLE_SORT,
                           DST_SYNC,
                           DST_ACCUM,
                           FUSED,
                           RANK_STAMPED,
                           TIE_ORDER,
                           TAG_BITS>,
                       DST_SYNC,
                       DST_ACCUM> {
    static void kernel(uint32_t m_iter, uint32_t k) {
        calculate_bitonic_topk_merge<
            APPROXIMATION_MODE,
            DST_ACCUM,
            IDIR,
            STABLE_SORT,
            FUSED,
            RANK_STAMPED,
            TIE_ORDER,
            TAG_BITS>(m_iter, k);
    }

    static void init_kernel() { topk_init<APPROXIMATION_MODE, FUSED, RANK_STAMPED, TAG_BITS>(); }
};

template <
    bool APPROXIMATION_MODE,
    bool STABLE_SORT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset,
    std::uint32_t TAG_BITS = 16>
struct TopkRebuild
    : SfpuUnaryOp<
          TopkRebuild<APPROXIMATION_MODE, STABLE_SORT, DST_SYNC, DST_ACCUM, FUSED, RANK_STAMPED, TIE_ORDER, TAG_BITS>,
          DST_SYNC,
          DST_ACCUM> {
    static void kernel(uint32_t idir, uint32_t m_iter, uint32_t k, uint32_t logk, uint32_t skip_second) {
        calculate_bitonic_topk_rebuild<APPROXIMATION_MODE, DST_ACCUM, STABLE_SORT, FUSED, RANK_STAMPED, TIE_ORDER>(
            idir, m_iter, k, logk, skip_second);
    }

    static void init_kernel() { topk_init<APPROXIMATION_MODE, FUSED, RANK_STAMPED, TAG_BITS>(); }
};

// Per-slab key sweeps for the fused-key / rank-stamped / comparator-stable topk modes (api/compute/topk.h).
// They run under the stage structs' topk_init and have no init of their own. API -> Struct<...>::calculate(args):
//   topk_fuse_tile                   -> TopkFuse<APPROX, LARGEST, ...>(idst, RC_custom)
//   topk_defuse_tile                 -> TopkDefuse<APPROX, LARGEST, INDEX_STORE_MODE, ...>(idst, RC_custom, n)
//   topk_stamp_local_positions       -> TopkStampLocalPositions<APPROX, LARGEST, TAG_BITS, ...>(idst, RC_custom)
//   topk_stamp_tile_rank_range       -> TopkStampTileRankRange<APPROX, LARGEST, TAG_BITS, ...>(idst, RC_custom, t, b)
//   topk_canonicalize_negzero_values -> TopkCanonicalizeNegzero<APPROX, ...>(idst, RC_custom)
template <bool APPROXIMATION_MODE, bool LARGEST, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkFuse : SfpuUnaryOp<TopkFuse<APPROXIMATION_MODE, LARGEST, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_topk_fuse<APPROXIMATION_MODE, LARGEST>(); }
};

template <bool APPROXIMATION_MODE, bool LARGEST, std::uint32_t INDEX_STORE_MODE, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkDefuse
    : SfpuUnaryOp<TopkDefuse<APPROXIMATION_MODE, LARGEST, INDEX_STORE_MODE, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t num_tiles) {
        calculate_topk_defuse<APPROXIMATION_MODE, LARGEST, INDEX_STORE_MODE>(num_tiles);
    }
};

template <bool APPROXIMATION_MODE, bool LARGEST, std::uint32_t TAG_BITS, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkStampLocalPositions
    : SfpuUnaryOp<
          TopkStampLocalPositions<APPROXIMATION_MODE, LARGEST, TAG_BITS, DST_SYNC, DST_ACCUM>,
          DST_SYNC,
          DST_ACCUM> {
    static void kernel() { calculate_topk_stamp_local_positions<APPROXIMATION_MODE, LARGEST, TAG_BITS>(); }
};

template <bool APPROXIMATION_MODE, bool LARGEST, std::uint32_t TAG_BITS, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkStampTileRankRange : SfpuUnaryOp<
                                    TopkStampTileRankRange<APPROXIMATION_MODE, LARGEST, TAG_BITS, DST_SYNC, DST_ACCUM>,
                                    DST_SYNC,
                                    DST_ACCUM> {
    static void kernel(uint32_t dst_tile_index, uint32_t rank_base) {
        calculate_topk_stamp_tile_rank_range<APPROXIMATION_MODE, LARGEST, TAG_BITS>(dst_tile_index, rank_base);
    }
};

template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM>
struct TopkCanonicalizeNegzero
    : SfpuUnaryOp<TopkCanonicalizeNegzero<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_topk_canonicalize_negzero<APPROXIMATION_MODE, DST_ACCUM>(); }
};

}  // namespace sfpu
}  // namespace ckernel
