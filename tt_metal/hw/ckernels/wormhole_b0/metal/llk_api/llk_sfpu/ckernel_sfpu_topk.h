// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false>
inline void calculate_bitonic_topk_phases_steps(
    std::uint32_t idir,
    std::uint32_t i_end_phase,
    std::uint32_t i_start_phase,
    std::uint32_t i_end_step,
    std::uint32_t i_start_step) {
    _bitonic_topk_phases_steps<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool idir = false,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    bool PRE_TAGGED = false>
inline void calculate_bitonic_topk_merge(std::uint32_t m_iter, std::uint32_t k) {
    _bitonic_topk_merge<APPROXIMATION_MODE, is_fp32_dest_acc_en, idir, STABLE_SORT, FUSED, RANK_STAMPED, PRE_TAGGED>(
        m_iter, k);
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false>
inline void calculate_bitonic_topk_rebuild(
    std::uint32_t idir, std::uint32_t m_iter, std::uint32_t k, std::uint32_t logk, std::uint32_t skip_second) {
    _bitonic_topk_rebuild<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED>(
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

// Rank-stamped stable topk stamp sweep (see _topk_stamp_local_positions_ in the LLK header;
// the companion strip `_topk_strip_rank_tags_` is called directly from the compute API, like
// _topk_uint16_move_dest_tile_to_pack_half_). The stamp runs once per freshly transposed
// 2-tile slab, before every local-sort call. largest is the op's GLOBAL sort order.
template <bool APPROXIMATION_MODE, bool largest>
inline void calculate_topk_stamp_local_positions() {
    _topk_stamp_local_positions_<largest>();
}

// Single-tile stamp with a runtime rank base: the k>32 insertion cascade's chain-position stamp
// (accumulator tile at level p gets [32p, 32p+32); the fresh chunk gets the top range once, at
// level 0; loser-tile tags ride). See _topk_stamp_tile_rank_range_ in the LLK header.
template <bool APPROXIMATION_MODE, bool largest>
inline void calculate_topk_stamp_tile_rank_range(std::uint32_t dst_tile_index, std::uint32_t rank_base) {
    _topk_stamp_tile_rank_range_<largest>(dst_tile_index, rank_base);
}

template <bool APPROXIMATION_MODE, bool FUSED = false, bool RANK_STAMPED = false>
inline void topk_init() {
    static_assert(!(FUSED && RANK_STAMPED), "fused and rank-stamped modes are mutually exclusive");
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 32}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (FUSED) {
        _init_topk_fused_();
    } else if constexpr (RANK_STAMPED) {
        _init_topk_rank_stamped_();
    } else {
        _init_topk();
    }
}

}  // namespace sfpu
}  // namespace ckernel
