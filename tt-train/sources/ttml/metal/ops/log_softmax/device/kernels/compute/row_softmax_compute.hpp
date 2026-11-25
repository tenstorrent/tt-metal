// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute steps shared by the row-softmax kernels (see row_softmax_program.hpp): the collapse-resistant row statistics
// of arXiv:2610.31642, Section 2, i.e. the row maximum M, the column K of its first maximum and the log-normalizer
// l = log1p(T), where the tail T excludes exactly x[K].
//
// A work unit is one tile row restricted to tile columns [begin, end). With `chunks` > 1
// the row is split across that many cores, each holding one unit; their partial statistics are exchanged
// through DRAM (see the dataflow kernels) and every core of the row combines them redundantly.
//
// Each step below is one function that acquires and releases DST itself, reserves and pushes every tile it writes,
// and waits for every tile it reads. A tile read by one step only is popped by that step; tiles that several steps
// read (the row statistics, the broadcast tiles) are popped by the function that runs those steps.
//
// Unpacker source-A format: the statistics buffers are all FP32; only the logits (cx) and the upstream gradient
// take their tensors' formats, BF16 or FP32 each. Between steps source A stays configured for FP32, so steps that
// copy statistics only never reconfigure it. A step that copies logits or gradients switches to their format and back
// with reconfig_data_format_srca(old, new), which reprograms the unpacker only when the two formats differ.

#pragma once

#include <limits>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "log_softmax_sfpu.h"

// Tile width from the host (the input's tensor_spec().tile()): absolute column = tile width * tile column + lane.
constexpr float kTileWidth = LOG_SOFTMAX_TILE_WIDTH;
// exp_tile mode for the exponentials outside the custom SFPU passes, which use their own FP32 exponential.
constexpr bool USE_APPROXIMATE_EXPONENTIAL = false;
// Circular buffers. Steps that use the program's fixed buffers take a type `Cbs` whose static constexpr members name
// them (see row_softmax_program.cpp): cx (the logits), ccols (lane columns), cm, cs, cic (per-lane statistics), ci
// (absolute columns), cd (lane offsets), cmax, cpivot, ctail, cnorm (row statistics), cpart0..cpart2 (the other cores'
// partials), cpartout (this core's partial) and cfp32 (any FP32 buffer: names the statistics format as the old or new
// operand of a source-A reconfiguration). Buffers that vary from call to call are template arguments of their own.

// Packs DST tiles 0..n-1 into n consecutive tiles of `cb`. The caller holds DST (after tile_regs_wait).
template <uint32_t cb>
inline void pack_n_tiles(uint32_t n) {
    cb_reserve_back(cb, n);
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < n; ++i) pack_tile(i, cb);
    cb_push_back(cb, n);
}
// Packs DST tile `reg` into one tile of `cb`. The caller holds DST (after tile_regs_wait).
template <uint32_t cb>
inline void pack_one_tile(uint32_t reg) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    cb_push_back(cb, 1);
}

// Writes column vector `index` of the FP32 buffer `cb` broadcast across all columns of a tile in `out`, so the
// per-tile passes need no column broadcasts of their own.
template <uint32_t cb, uint32_t out>
inline void broadcast_column_tile(uint32_t index) {
    cb_wait_front(cb, index + 1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.F);  // dst0 = 0
    copy_init(cb);
    copy_tile(cb, index, 1);  // dst1 = v, column vector `index` of cb
    sfpu_bcast_col_init();
    sfpu_add_bcast_col(0, 1);  // dst0 = v in every column
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<out>(0);  // out = v broadcast
    tile_regs_release();
}

// The four steps of combine_lanes (below). Per lane: m (cb_m), i (cb_i), t (cb_t); per row: M, K, T.

// M = max over lanes of m: cb_m -> cmax.
template <typename Cbs, uint32_t cb_m>
inline void lane_maximum_tile() {
    cb_wait_front(cb_m, 1);
    tile_regs_acquire();
    copy_init(cb_m);
    copy_tile(cb_m, 0, 0);  // dst0 = m
    sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
    sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = M = max over lanes of m
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cbs::cmax>(0);  // cmax = M
    tile_regs_release();
}

// d = m - M per lane, kept at -inf for an empty lane: cb_m, cmax -> cd.
template <typename Cbs, uint32_t cb_m>
inline void lane_offset_tile() {
    cb_wait_front(Cbs::cmax, 1);
    tile_regs_acquire();
    copy_init(cb_m);
    copy_tile(cb_m, 0, 0);  // dst0 = m
    copy_init(Cbs::cmax);
    copy_tile(Cbs::cmax, 0, 1);  // dst1 = M
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = d = m - M
    copy_init(cb_m);
    copy_tile(cb_m, 0, 2);                                                // dst2 = m
    log_softmax_mask<LogSoftmaxMask::EmptyLane, /*index_slot*/ 2>(0, 0);  // d = -inf where m = -inf
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cbs::cd>(0);  // cd = d
    tile_regs_release();
}

// K = min of i over the lanes with d == 0: cd, cb_i -> cpivot.
template <typename Cbs, uint32_t cb_i>
inline void first_maximum_column_tile() {
    cb_wait_front(Cbs::cd, 1);
    cb_wait_front(cb_i, 1);
    tile_regs_acquire();
    copy_init(Cbs::cd);
    copy_tile(Cbs::cd, 0, 0);  // dst0 = d
    copy_init(cb_i);
    copy_tile(cb_i, 0, 1);                                                               // dst1 = i
    log_softmax_mask<LogSoftmaxMask::FirstMaximumIndex, /*index_slot*/ 1>(0, 1U << 25);  // dst0 = d == 0 ? i : +inf
    sfpu_reduce_init<PoolType::MIN, DataFormat::Float32>();
    sfpu_reduce<PoolType::MIN, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = K = min of those i
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cbs::cpivot>(0);  // cpivot = K
    tile_regs_release();
}

// T = sum over lanes of exp(d) * t + exp(d), without the selected lane's exp(d) = 1: cd, cb_t, cb_i, cpivot -> ctail.
template <typename Cbs, uint32_t cb_i, uint32_t cb_t>
inline void combined_tail_tile() {
    cb_wait_front(cb_t, 1);
    cb_wait_front(Cbs::cpivot, 1);
    tile_regs_acquire();
    copy_init(Cbs::cd);
    copy_tile(Cbs::cd, 0, 0);  // dst0 = d
    exp_tile_init<USE_APPROXIMATE_EXPONENTIAL>();
    exp_tile<USE_APPROXIMATE_EXPONENTIAL>(0);  // dst0 = exp(d)
    copy_init(cb_t);
    copy_tile(cb_t, 0, 1);  // dst1 = t
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 1);  // dst1 = t * exp(d)
    copy_init(cb_i);
    copy_tile(cb_i, 0, 2);  // dst2 = i
    copy_init(Cbs::cpivot);
    copy_tile(Cbs::cpivot, 0, 3);  // dst3 = K
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(2, 3);  // dst2 = i - K
    // Never add the selected exp(0) = 1 to the tail (arXiv:2610.31642, Eq. 5).
    log_softmax_mask<LogSoftmaxMask::ExcludeSelected, /*index_slot*/ 2>(0, 0);  // dst0 = 0 where i == K
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dst0 = [i != K] exp(d) + t * exp(d)
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = T = sum over lanes
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cbs::ctail>(0);  // ctail = T
    tile_regs_release();
}

// Merges partial softmax statistics of disjoint parts of each row into the statistics of the whole row.
//
// Inputs: three column-vector tiles whose lanes (tile columns) each describe one part of the row:
//   cb_m: m, the part's maximum;
//   cb_i: i, the absolute column of the part's first maximum;
//   cb_t: t, the part's tail: sum of exp(x - m) over the part, except that first maximum.
// The parts are
//   - the SFPU lanes of one core, after its pass over its tile columns;
//   - the cores sharing a row (chunks > 1), one per lane, after the exchange through DRAM.
// The inputs are consumed (popped).
//
// Outputs, one value per row (a column vector), each pushed to its buffer:
//   cmax:   M, the row maximum;
//   cpivot: K, the column of the row's first maximum (the lowest column among the parts holding M);
//   ctail:  T, the row's tail: sum of exp(x - M) over the row, except exactly x[K].
// log1p(T) is then the log-normalizer, accurate however small T is (see log_normalizer_tile).
//
// With d = m - M per lane:  T = sum over lanes of t * exp(d) + [i != K] exp(d).
// An empty part (m = -inf, i = +inf, t = 0: it saw only -inf, or is unused) contributes nothing, even when
// every part is empty, as for a core or shard whose columns of a row are all masked.
template <typename Cbs, uint32_t cb_m, uint32_t cb_i, uint32_t cb_t>
inline void combine_lanes() {
    lane_maximum_tile<Cbs, cb_m>();          // cmax = M
    lane_offset_tile<Cbs, cb_m>();           // cd = d
    first_maximum_column_tile<Cbs, cb_i>();  // cpivot = K
    combined_tail_tile<Cbs, cb_i, cb_t>();   // ctail = T
    cb_pop_front(Cbs::cd, 1);
    cb_pop_front(cb_m, 1);
    cb_pop_front(cb_i, 1);
    cb_pop_front(cb_t, 1);
}

// l = log1p(T), accurate however small the tail (Eq. 6): ctail -> cnorm. Pops ctail.
template <typename Cbs>
inline void log_normalizer_tile() {
    cb_wait_front(Cbs::ctail, 1);
    tile_regs_acquire();
    copy_init(Cbs::ctail);
    copy_tile(Cbs::ctail, 0, 0);  // dst0 = T
    log1p_tile_init</*fast_and_approx*/ false>();
    log1p_tile</*fast_and_approx*/ false>(0);  // dst0 = l = log1p(T)
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cbs::cnorm>(0);  // cnorm = l
    tile_regs_release();
    cb_pop_front(Cbs::ctail, 1);
}

// Pass 1 over this unit's logits, online per lane: the running maximum m, the tail t (sum of exp(x - m) except one
// first maximum) and the tile column c of that first maximum: cx -> cm, cs, cic. Pops each logits tile.
template <typename Cbs>
inline void lane_statistics_tiles(uint32_t begin, uint32_t end) {
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, -std::numeric_limits<float>::infinity());  // m = -inf: the lanes' running maximum
    fill_tile(1, 0.F);                                      // t = 0: their tail
    fill_tile(2, std::numeric_limits<float>::infinity());   // c = +inf: the tile column of their first maximum
    reconfig_data_format_srca(Cbs::cfp32, Cbs::cx);
    copy_init(Cbs::cx);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(Cbs::cx, 1);
        copy_tile(Cbs::cx, 0, 3);    // dst3 = x, the next tile of logits
        log_softmax_online(0, col);  // update m, t and c with x
        cb_pop_front(Cbs::cx, 1);
    }
    reconfig_data_format_srca(Cbs::cx, Cbs::cfp32);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(Cbs::cm, 1);
    cb_reserve_back(Cbs::cs, 1);
    cb_reserve_back(Cbs::cic, 1);
    pack_reconfig_data_format(Cbs::cm);
    pack_tile(0, Cbs::cm);   // cm = m
    pack_tile(1, Cbs::cs);   // cs = t
    pack_tile(2, Cbs::cic);  // cic = c
    cb_push_back(Cbs::cm, 1);
    cb_push_back(Cbs::cs, 1);
    cb_push_back(Cbs::cic, 1);
    tile_regs_release();
}

// i = c * tile width + lane column, the absolute column of each lane's first maximum: cic, ccols -> ci. Pops cic.
template <typename Cbs>
inline void absolute_column_tile() {
    cb_wait_front(Cbs::cic, 1);
    tile_regs_acquire();
    copy_init(Cbs::cic);
    copy_tile(Cbs::cic, 0, 0);  // dst0 = c
    binop_with_scalar_tile_init();
    mul_unary_tile(0, __builtin_bit_cast(uint32_t, kTileWidth));  // dst0 = c * tile width
    copy_init(Cbs::ccols);
    copy_tile(Cbs::ccols, 0, 1);  // dst1 = lane column, 0 .. tile width - 1
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dst0 = i, the absolute column
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cbs::ci>(0);  // ci = i
    tile_regs_release();
    cb_pop_front(Cbs::cic, 1);
}

// This core's (M, K, T), published for the other cores of the row: cmax, cpivot, ctail -> cpartout. Pops the three.
template <typename Cbs>
inline void partial_statistics_tiles() {
    cb_wait_front(Cbs::cmax, 1);
    cb_wait_front(Cbs::cpivot, 1);
    cb_wait_front(Cbs::ctail, 1);
    tile_regs_acquire();
    copy_init(Cbs::cmax);
    copy_tile(Cbs::cmax, 0, 0);  // dst0 = M
    copy_init(Cbs::cpivot);
    copy_tile(Cbs::cpivot, 0, 1);  // dst1 = K
    copy_init(Cbs::ctail);
    copy_tile(Cbs::ctail, 0, 2);  // dst2 = T
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles<Cbs::cpartout>(3);  // cpartout = this core's (M, K, T)
    tile_regs_release();
    cb_pop_front(Cbs::cmax, 1);
    cb_pop_front(Cbs::cpivot, 1);
    cb_pop_front(Cbs::ctail, 1);
}

// The row's statistics from this unit's logits, with `chunks` cores sharing the row: M (cmax), K (cpivot) and
// l = log1p(T) (cnorm).
template <typename Cbs, uint32_t chunks>
inline void row_statistics(uint32_t begin, uint32_t end) {
    lane_statistics_tiles<Cbs>(begin, end);           // cm, cs, cic = m, t, c per lane
    absolute_column_tile<Cbs>();                      // ci = i
    combine_lanes<Cbs, Cbs::cm, Cbs::ci, Cbs::cs>();  // cmax, cpivot, ctail = M, K, T of this core's columns
    if constexpr (chunks > 1) {
        partial_statistics_tiles<Cbs>();                              // cpartout = this core's (M, K, T)
        combine_lanes<Cbs, Cbs::cpart0, Cbs::cpart1, Cbs::cpart2>();  // cmax, cpivot, ctail = M, K, T of the whole row
    }
    log_normalizer_tile<Cbs>();  // cnorm = l = log1p(T)
}
