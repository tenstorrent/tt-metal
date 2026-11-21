// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

// Shared by the fused log_softmax and, with CROSS_ENTROPY, the fused cross-entropy (see row_softmax_program.hpp).
// A work unit is one tile row restricted to tile columns [begin, end). With `chunks` > 1
// the row is split across that many cores, each holding one unit; their partial statistics are exchanged
// through DRAM (see the dataflow kernels) and every core of the row combines them redundantly.
//
// Each step below is one function that acquires and releases DST itself, reserves and pushes every tile it writes,
// and waits for every tile it reads. A tile read by one step only is popped by that step; tiles that several steps
// read (the row statistics, the broadcast tiles) are popped by the function that runs those steps.
//
// Unpacker source-A format: the statistics buffers are all FP32; only the logits (cx) and the upstream gradient (cdy)
// take their tensors' formats, BF16 or FP32 each. Between steps source A stays configured for FP32, so steps that
// copy statistics only never reconfigure it. A step that copies logits or gradients switches to their format and back
// with reconfig_data_format_srca(old, new), which reprograms the unpacker only when the two formats differ.
constexpr uint32_t chunks = get_compile_time_arg_val(0);
// Tile width from the host (the input's tensor_spec().tile()): absolute column = tile width * tile column + lane.
constexpr float kTileWidth = LOG_SOFTMAX_TILE_WIDTH;
// exp_tile mode for the exponentials outside the custom SFPU passes, which use their own FP32 exponential.
constexpr bool USE_APPROXIMATE_EXPONENTIAL = false;
constexpr uint32_t scaler_bits = get_compile_time_arg_val(1);  // Cross-entropy backward: s.
constexpr uint32_t cx = 0, cdy = 1, ccols = 2, cm = 3, cs = 4, cic = 5, ci = 6, cmax = 7, cstateout = 8, cstatein = 9,
                   cpivot = 10, ctail = 11, cd = 12, cnorm = 13, csum = 14, cothers = 15, cout = 16, cpart0 = 17,
                   cpart1 = 18, cpart2 = 19, cpivotdy = 20, cpivotgrad = 21, cpartout = 23, ctargetx = 25,
                   cfirstcol = 27, cupstream = 29;
// Any FP32 buffer: names the statistics format as the old or new operand of a source-A reconfiguration.
constexpr uint32_t cfp32 = ccols;

// Packs DST tiles 0..n-1 into n consecutive tiles of `cb`. The caller holds DST (after tile_regs_wait).
inline void pack_n_tiles(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < n; ++i) pack_tile(i, cb);
    cb_push_back(cb, n);
}
// Packs DST tile `reg` into one tile of `cb`. The caller holds DST (after tile_regs_wait).
inline void pack_one_tile(uint32_t reg, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    cb_push_back(cb, 1);
}

// Writes column vector `index` of the FP32 buffer `cb` broadcast across all columns of a tile in `out`, so the
// per-tile passes need no column broadcasts of their own.
inline void broadcast_column_tile(uint32_t cb, uint32_t index, uint32_t out) {
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
    pack_one_tile(0, out);  // out = v broadcast
    tile_regs_release();
}

#if !defined(BACKWARD) || defined(CROSS_ENTROPY)
// The four steps of combine_lanes (below). Per lane: m (cb_m), i (cb_i), t (cb_t); per row: M, K, T.

// M = max over lanes of m: cb_m -> cmax.
inline void lane_maximum_tile(uint32_t cb_m) {
    cb_wait_front(cb_m, 1);
    tile_regs_acquire();
    copy_init(cb_m);
    copy_tile(cb_m, 0, 0);  // dst0 = m
    sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
    sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = M = max over lanes of m
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cmax);  // cmax = M
    tile_regs_release();
}

// d = m - M per lane, kept at -inf for an empty lane: cb_m, cmax -> cd.
inline void lane_offset_tile(uint32_t cb_m) {
    cb_wait_front(cmax, 1);
    tile_regs_acquire();
    copy_init(cb_m);
    copy_tile(cb_m, 0, 0);  // dst0 = m
    copy_init(cmax);
    copy_tile(cmax, 0, 1);  // dst1 = M
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = d = m - M
    copy_init(cb_m);
    copy_tile(cb_m, 0, 2);                                                // dst2 = m
    log_softmax_mask<LogSoftmaxMask::EmptyLane, /*index_slot*/ 2>(0, 0);  // d = -inf where m = -inf
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cd);  // cd = d
    tile_regs_release();
}

// K = min of i over the lanes with d == 0: cd, cb_i -> cpivot.
inline void first_maximum_column_tile(uint32_t cb_i) {
    cb_wait_front(cd, 1);
    cb_wait_front(cb_i, 1);
    tile_regs_acquire();
    copy_init(cd);
    copy_tile(cd, 0, 0);  // dst0 = d
    copy_init(cb_i);
    copy_tile(cb_i, 0, 1);                                                               // dst1 = i
    log_softmax_mask<LogSoftmaxMask::FirstMaximumIndex, /*index_slot*/ 1>(0, 1U << 25);  // dst0 = d == 0 ? i : +inf
    sfpu_reduce_init<PoolType::MIN, DataFormat::Float32>();
    sfpu_reduce<PoolType::MIN, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = K = min of those i
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cpivot);  // cpivot = K
    tile_regs_release();
}

// T = sum over lanes of exp(d) * t + exp(d), without the selected lane's exp(d) = 1: cd, cb_t, cb_i, cpivot -> ctail.
inline void combined_tail_tile(uint32_t cb_i, uint32_t cb_t) {
    cb_wait_front(cb_t, 1);
    cb_wait_front(cpivot, 1);
    tile_regs_acquire();
    copy_init(cd);
    copy_tile(cd, 0, 0);  // dst0 = d
    exp_tile_init<USE_APPROXIMATE_EXPONENTIAL>();
    exp_tile<USE_APPROXIMATE_EXPONENTIAL>(0);  // dst0 = exp(d)
    copy_init(cb_t);
    copy_tile(cb_t, 0, 1);  // dst1 = t
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 1);  // dst1 = t * exp(d)
    copy_init(cb_i);
    copy_tile(cb_i, 0, 2);  // dst2 = i
    copy_init(cpivot);
    copy_tile(cpivot, 0, 3);  // dst3 = K
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
    pack_one_tile(0, ctail);  // ctail = T
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
inline void combine_lanes(uint32_t cb_m, uint32_t cb_i, uint32_t cb_t) {
    lane_maximum_tile(cb_m);          // cmax = M
    lane_offset_tile(cb_m);           // cd = d
    first_maximum_column_tile(cb_i);  // cpivot = K
    combined_tail_tile(cb_i, cb_t);   // ctail = T
    cb_pop_front(cd, 1);
    cb_pop_front(cb_m, 1);
    cb_pop_front(cb_i, 1);
    cb_pop_front(cb_t, 1);
}

// l = log1p(T), accurate however small the tail (Eq. 6): ctail -> cnorm. Pops ctail.
inline void log_normalizer_tile() {
    cb_wait_front(ctail, 1);
    tile_regs_acquire();
    copy_init(ctail);
    copy_tile(ctail, 0, 0);  // dst0 = T
    log1p_tile_init</*fast_and_approx*/ false>();
    log1p_tile</*fast_and_approx*/ false>(0);  // dst0 = l = log1p(T)
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cnorm);  // cnorm = l
    tile_regs_release();
    cb_pop_front(ctail, 1);
}

// Pass 1 over this unit's logits, online per lane: the running maximum m, the tail t (sum of exp(x - m) except one
// first maximum) and the tile column c of that first maximum: cx -> cm, cs, cic. Pops each logits tile.
inline void lane_statistics_tiles(uint32_t begin, uint32_t end) {
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, -std::numeric_limits<float>::infinity());  // m = -inf: the lanes' running maximum
    fill_tile(1, 0.F);                                      // t = 0: their tail
    fill_tile(2, std::numeric_limits<float>::infinity());   // c = +inf: the tile column of their first maximum
    reconfig_data_format_srca(cfp32, cx);
    copy_init(cx);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(cx, 1);
        copy_tile(cx, 0, 3);         // dst3 = x, the next tile of logits
        log_softmax_online(0, col);  // update m, t and c with x
        cb_pop_front(cx, 1);
    }
    reconfig_data_format_srca(cx, cfp32);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cm, 1);
    cb_reserve_back(cs, 1);
    cb_reserve_back(cic, 1);
    pack_reconfig_data_format(cm);
    pack_tile(0, cm);   // cm = m
    pack_tile(1, cs);   // cs = t
    pack_tile(2, cic);  // cic = c
    cb_push_back(cm, 1);
    cb_push_back(cs, 1);
    cb_push_back(cic, 1);
    tile_regs_release();
}

// i = c * tile width + lane column, the absolute column of each lane's first maximum: cic, ccols -> ci. Pops cic.
inline void absolute_column_tile() {
    cb_wait_front(cic, 1);
    tile_regs_acquire();
    copy_init(cic);
    copy_tile(cic, 0, 0);  // dst0 = c
    binop_with_scalar_tile_init();
    mul_unary_tile(0, __builtin_bit_cast(uint32_t, kTileWidth));  // dst0 = c * tile width
    copy_init(ccols);
    copy_tile(ccols, 0, 1);  // dst1 = lane column, 0 .. tile width - 1
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dst0 = i, the absolute column
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, ci);  // ci = i
    tile_regs_release();
    cb_pop_front(cic, 1);
}

// This core's (M, K, T), published for the other cores of the row: cmax, cpivot, ctail -> cpartout. Pops the three.
inline void partial_statistics_tiles() {
    cb_wait_front(cmax, 1);
    cb_wait_front(cpivot, 1);
    cb_wait_front(ctail, 1);
    tile_regs_acquire();
    copy_init(cmax);
    copy_tile(cmax, 0, 0);  // dst0 = M
    copy_init(cpivot);
    copy_tile(cpivot, 0, 1);  // dst1 = K
    copy_init(ctail);
    copy_tile(ctail, 0, 2);  // dst2 = T
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles(cpartout, 3);  // cpartout = this core's (M, K, T)
    tile_regs_release();
    cb_pop_front(cmax, 1);
    cb_pop_front(cpivot, 1);
    cb_pop_front(ctail, 1);
}

// The row's statistics from this unit's logits: M (cmax), K (cpivot) and l = log1p(T) (cnorm).
inline void row_statistics(uint32_t begin, uint32_t end) {
    lane_statistics_tiles(begin, end);  // cm, cs, cic = m, t, c per lane
    absolute_column_tile();             // ci = i
    combine_lanes(cm, ci, cs);          // cmax, cpivot, ctail = M, K, T of this core's columns
    if constexpr (chunks > 1) {
        partial_statistics_tiles();             // cpartout = this core's (M, K, T)
        combine_lanes(cpart0, cpart1, cpart2);  // cmax, cpivot, ctail = M, K, T of the whole row
    }
    log_normalizer_tile();  // cnorm = l = log1p(T)
}
#endif

#ifdef CROSS_ENTROPY
// Loss of each row from its target logit x_t (column 0 of ctargetx; zero for an out-of-range target):
// -y_t with y_t = (x_t - M) - l, computed exactly as the log_softmax output. When the target is
// the selected maximum, x_t - M = 0 and the loss is l = log1p(T) itself, however small.
// ctargetx, cmax, cnorm, cfirstcol -> cout. Pops ctargetx and the row statistics cmax, cpivot, cnorm.
inline void cross_entropy_loss_tile() {
    cb_wait_front(ctargetx, 1);
    cb_wait_front(cmax, 1);
    cb_wait_front(cnorm, 1);
    tile_regs_acquire();
    copy_init(ctargetx);
    copy_tile(ctargetx, 0, 0);  // dst0 = x_t
    copy_init(cmax);
    copy_tile(cmax, 0, 1);  // dst1 = M
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = x_t - M
    copy_init(cnorm);
    copy_tile(cnorm, 0, 1);  // dst1 = l
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = y_t = (x_t - M) - l
    negative_tile_init();
    negative_tile(0);  // dst0 = loss = -y_t
    // Keep column 0 only, so the loss tile's padding columns are zero.
    copy_init(cfirstcol);
    copy_tile(cfirstcol, 0, 1);  // dst1 = 1 in column 0, 0 elsewhere
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);  // dst0 = loss in column 0 only
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cout);  // cout = loss
    tile_regs_release();
    cb_pop_front(ctargetx, 1);
    cb_pop_front(cmax, 1);
    cb_pop_front(cpivot, 1);
    cb_pop_front(cnorm, 1);
}

#ifdef UPSTREAM_GRAD
// Multiplies the column vector in DST tile `dst` by each row's upstream gradient g (cupstream, column 0), using DST
// tile dst + 1. Part of a step: the caller holds DST.
inline void multiply_by_upstream_tile(uint32_t dst) {
    copy_init(cupstream);
    copy_tile(cupstream, 0, dst + 1);  // dst[dst + 1] = g, the upstream gradient
    mul_binary_tile_init();
    mul_binary_tile(dst, dst + 1, dst);  // dst[dst] *= g
}
#endif

// q = s * exp(-l) (times g with UPSTREAM_GRAD), so that dx = exp(x - M) * q = s * softmax(x): cnorm -> cd.
inline void cross_entropy_scale_tile() {
    cb_wait_front(cnorm, 1);
#ifdef UPSTREAM_GRAD
    cb_wait_front(cupstream, 1);
#endif
    tile_regs_acquire();
    copy_init(cnorm);
    copy_tile(cnorm, 0, 0);  // dst0 = l
    negative_tile_init();
    negative_tile(0);  // dst0 = -l
    exp_tile_init<USE_APPROXIMATE_EXPONENTIAL>();
    exp_tile<USE_APPROXIMATE_EXPONENTIAL>(0);  // dst0 = exp(-l) = 1 / (1 + T)
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scaler_bits);  // dst0 = q = s * exp(-l)
#ifdef UPSTREAM_GRAD
    multiply_by_upstream_tile(0);  // dst0 = q * g
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cd);  // cd = q
    tile_regs_release();
}

// The target's gradient dx_t = s * (p_t - 1) = s * expm1(y_t) (times g), y_t = (x_t - M) - l: for a target at the
// selected maximum that is -s * T / (1 + T), the sum of the other gradients (arXiv:2610.31642, Eq. 7), without
// subtracting nearly equal values. ctargetx, cmax, cnorm -> cpivotgrad, which the writer substitutes. Pops ctargetx
// (and cupstream).
inline void target_gradient_tile() {
    cb_wait_front(ctargetx, 1);
    cb_wait_front(cmax, 1);
    tile_regs_acquire();
    copy_init(ctargetx);
    copy_tile(ctargetx, 0, 0);  // dst0 = x_t
    copy_init(cmax);
    copy_tile(cmax, 0, 1);  // dst1 = M
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = x_t - M
    copy_init(cnorm);
    copy_tile(cnorm, 0, 1);  // dst1 = l
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(0, 1);  // dst0 = y_t = (x_t - M) - l
    expm1_tile_init();
    expm1_tile(0);  // dst0 = p_t - 1 = expm1(y_t)
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scaler_bits);  // dst0 = s * (p_t - 1)
#ifdef UPSTREAM_GRAD
    multiply_by_upstream_tile(0);  // dst0 = s * (p_t - 1) * g
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cpivotgrad);  // cpivotgrad = dx_t, the target's gradient
    tile_regs_release();
    cb_pop_front(ctargetx, 1);
#ifdef UPSTREAM_GRAD
    cb_pop_front(cupstream, 1);
#endif
}

// dx = exp(x - M) * q over this unit's logits, with M (cm) and q (cs) broadcast across the row: cx -> cout. Pops
// each logits tile.
inline void cross_entropy_grad_tiles(uint32_t begin, uint32_t end) {
    cb_wait_front(cm, 1);
    cb_wait_front(cs, 1);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(cx, 1);
        tile_regs_acquire();
        copy_init(cm);
        copy_tile(cm, 0, 1);  // dst1 = M
        copy_init(cs);
        copy_tile(cs, 0, 2);  // dst2 = q
        reconfig_data_format_srca(cfp32, cx);
        copy_init(cx);
        copy_tile(cx, 0, 0);  // dst0 = x
        reconfig_data_format_srca(cx, cfp32);
        cross_entropy_grad(0);  // dst0 = dx = exp(x - M) * q
        tile_regs_commit();
        tile_regs_wait();
        pack_one_tile(0, cout);  // cout = dx
        tile_regs_release();
        cb_pop_front(cx, 1);
    }
}

// One row of the cross-entropy gradient: dx = s * softmax(x), with the writer substituting the target's dx_t.
// Consumes the row statistics cmax, cpivot, cnorm.
inline void cross_entropy_grad_row(uint32_t begin, uint32_t end) {
    cross_entropy_scale_tile();        // cd = q
    broadcast_column_tile(cd, 0, cs);  // cs = q in every column
    cb_pop_front(cd, 1);
    target_gradient_tile();              // cpivotgrad = dx_t
    broadcast_column_tile(cmax, 0, cm);  // cm = M in every column
    cb_pop_front(cmax, 1);
    cb_pop_front(cpivot, 1);
    cb_pop_front(cnorm, 1);
    cross_entropy_grad_tiles(begin, end);  // cout = dx
    cb_pop_front(cm, 1);
    cb_pop_front(cs, 1);
}
#endif

#if !defined(BACKWARD) && !defined(CROSS_ENTROPY)
// The row state (M, K, l) for the backward: cmax, cpivot, cnorm -> cstateout.
inline void log_softmax_row_state_tiles() {
    cb_wait_front(cmax, 1);
    cb_wait_front(cpivot, 1);
    cb_wait_front(cnorm, 1);
    tile_regs_acquire();
    copy_init(cmax);
    copy_tile(cmax, 0, 0);  // dst0 = M
    copy_init(cpivot);
    copy_tile(cpivot, 0, 1);  // dst1 = K
    copy_init(cnorm);
    copy_tile(cnorm, 0, 2);  // dst2 = l
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles(cstateout, 3);  // cstateout = row state (M, K, l) for the backward
    tile_regs_release();
}

// Pass 2, native-precision output y = (x - M) - l over this unit's logits, with M (cm) and l (cs) broadcast across
// the row: cx -> cout. Pops each logits tile.
inline void log_softmax_output_tiles(uint32_t begin, uint32_t end) {
    cb_wait_front(cm, 1);
    cb_wait_front(cs, 1);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(cx, 1);
        tile_regs_acquire();
        copy_init(cm);
        copy_tile(cm, 0, 1);  // dst1 = M
        copy_init(cs);
        copy_tile(cs, 0, 2);  // dst2 = l
        reconfig_data_format_srca(cfp32, cx);
        copy_init(cx);
        copy_tile(cx, 0, 0);  // dst0 = x
        reconfig_data_format_srca(cx, cfp32);
        log_softmax_output(0);  // dst0 = y = (x - M) - l
        tile_regs_commit();
        tile_regs_wait();
        pack_one_tile(0, cout);  // cout = y
        tile_regs_release();
        cb_pop_front(cx, 1);
    }
}

// One row of the log_softmax forward: its statistics, the row state for the backward, then the output.
inline void log_softmax_forward_row(uint32_t begin, uint32_t end) {
    row_statistics(begin, end);     // cmax, cpivot, cnorm = M, K, l
    log_softmax_row_state_tiles();  // cstateout = (M, K, l)
    cb_pop_front(cpivot, 1);
    broadcast_column_tile(cmax, 0, cm);   // cm = M in every column
    broadcast_column_tile(cnorm, 0, cs);  // cs = l in every column
    cb_pop_front(cmax, 1);
    cb_pop_front(cnorm, 1);
    log_softmax_output_tiles(begin, end);  // cout = y
    cb_pop_front(cm, 1);
    cb_pop_front(cs, 1);
}
#endif

#if defined(BACKWARD) && !defined(CROSS_ENTROPY)
// Pass 1 of the backward, per lane and then per row: A, the sum of the upstream gradient over this unit's columns
// except the selected element (the reader zeroes it and hands it over separately in cpivotdy): cdy -> cothers. Pops
// each gradient tile.
inline void upstream_sum_tiles(uint32_t begin, uint32_t end) {
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.F);  // dst0 = 0: the lanes' running sum of dy
    reconfig_data_format_srca(cfp32, cdy);
    copy_init(cdy);
    add_binary_tile_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(cdy, 1);
        copy_tile(cdy, 0, 1);      // dst1 = dy
        add_binary_tile(0, 1, 0);  // dst0 += dy
        cb_pop_front(cdy, 1);
    }
    reconfig_data_format_srca(cdy, cfp32);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = A: sum of dy without dy_k
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cothers);  // cothers = A
    tile_regs_release();
}

// This core's (A, dy_k), published for the other cores of the row: cothers, cpivotdy -> cpartout. Pops the two.
inline void partial_upstream_sums_tiles() {
    cb_wait_front(cothers, 1);
    cb_wait_front(cpivotdy, 1);
    tile_regs_acquire();
    copy_init(cothers);
    copy_tile(cothers, 0, 0);  // dst0 = A
    copy_init(cpivotdy);
    copy_tile(cpivotdy, 0, 1);  // dst1 = dy_k, or 0 when another core holds k
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles(cpartout, 2);  // cpartout = this core's (A, dy_k)
    tile_regs_release();
    cb_pop_front(cothers, 1);
    cb_pop_front(cpivotdy, 1);
}

// The sum over lanes (the cores of the row) of each row of `cb_in` -> `cb_out`. Pops cb_in.
inline void sum_over_lanes_tile(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_init(cb_in);
    copy_tile(cb_in, 0, 0);  // dst0 = one value per core, one per lane
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = their sum
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cb_out);  // cb_out = the sum
    tile_regs_release();
    cb_pop_front(cb_in, 1);
}

// S = A + dy_k, the row's full sum of the upstream gradient: cothers, pivot_dy -> csum.
inline void upstream_total_tile(uint32_t pivot_dy) {
    cb_wait_front(cothers, 1);
    cb_wait_front(pivot_dy, 1);
    tile_regs_acquire();
    copy_init(cothers);
    copy_tile(cothers, 0, 0);  // dst0 = A
    copy_init(pivot_dy);
    copy_tile(pivot_dy, 0, 1);  // dst1 = dy_k
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dst0 = S = A + dy_k
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, csum);  // csum = S
    tile_regs_release();
}

// Eq. (7) in closed form, the selected element's gradient dx_k = -sum_{j!=k} (dy_j - p_j S) = S * q' - A with
// q' = sum_{j!=k} p_j = T / (1 + T) = -expm1(-log1p(T)), accurate when p_k rounds to one:
// cstatein (l), csum, cothers -> cpivotgrad, which the writer substitutes.
inline void selected_gradient_tile() {
    cb_wait_front(csum, 1);
    tile_regs_acquire();
    copy_init(cstatein);
    copy_tile(cstatein, 2, 0);  // dst0 = l
    negative_tile_init();
    negative_tile(0);  // dst0 = -l
    expm1_tile_init();
    expm1_tile(0);  // dst0 = expm1(-l) = -T / (1 + T)
    copy_init(csum);
    copy_tile(csum, 0, 1);  // dst1 = S
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);  // dst0 = -S * T / (1 + T)
    copy_init(cothers);
    copy_tile(cothers, 0, 1);  // dst1 = A
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dst0 = A - S * T / (1 + T)
    negative_tile_init();
    negative_tile(0);  // dst0 = dx_k = S * T / (1 + T) - A
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cpivotgrad);  // cpivotgrad = dx_k
    tile_regs_release();
}

// q = S * exp(-l) = S * p_k, so that softmax(x) * S = exp(x - M) * q: cstatein (l), csum -> cnorm.
inline void backward_scale_tile() {
    tile_regs_acquire();
    copy_init(cstatein);
    copy_tile(cstatein, 2, 0);  // dst0 = l
    negative_tile_init();
    negative_tile(0);  // dst0 = -l
    exp_tile_init<USE_APPROXIMATE_EXPONENTIAL>();
    exp_tile<USE_APPROXIMATE_EXPONENTIAL>(0);  // dst0 = p_k = exp(-l)
    copy_init(csum);
    copy_tile(csum, 0, 1);  // dst1 = S
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);  // dst0 = q = S * p_k
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile(0, cnorm);  // cnorm = q
    tile_regs_release();
}

// Pass 2, dx = dy - exp(x - M) * q over this unit's columns, with M (cm) and q (cs) broadcast across the row:
// cx, cdy -> cout. Pops each logits and gradient tile. The writer substitutes dx_k.
inline void log_softmax_grad_tiles(uint32_t begin, uint32_t end) {
    cb_wait_front(cm, 1);
    cb_wait_front(cs, 1);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(cx, 1);
        cb_wait_front(cdy, 1);
        tile_regs_acquire();
        copy_init(cm);
        copy_tile(cm, 0, 2);  // dst2 = M
        copy_init(cs);
        copy_tile(cs, 0, 3);  // dst3 = q
        reconfig_data_format_srca(cfp32, cx);
        copy_init(cx);
        copy_tile(cx, 0, 0);  // dst0 = x
        reconfig_data_format_srca(cx, cdy);
        copy_init(cdy);
        copy_tile(cdy, 0, 1);  // dst1 = dy
        reconfig_data_format_srca(cdy, cfp32);
        log_softmax_grad(0);  // dst0 = dx = dy - exp(x - M) * q
        tile_regs_commit();
        tile_regs_wait();
        pack_one_tile(0, cout);  // cout = dx
        tile_regs_release();
        cb_pop_front(cx, 1);
        cb_pop_front(cdy, 1);
    }
}

// One row of the log_softmax backward, from the row state (M, K, l) in cstatein.
inline void log_softmax_backward_row(uint32_t begin, uint32_t end) {
    cb_wait_front(cstatein, 3);
    upstream_sum_tiles(begin, end);  // cothers = A
    uint32_t pivot_dy = cpivotdy;
    if constexpr (chunks > 1) {
        partial_upstream_sums_tiles();         // cpartout = this core's (A, dy_k)
        sum_over_lanes_tile(cpart0, cothers);  // cothers = A of the whole row
        sum_over_lanes_tile(cpart1, cd);       // cd = dy_k
        pivot_dy = cd;
    }
    upstream_total_tile(pivot_dy);  // csum = S
    selected_gradient_tile();       // cpivotgrad = dx_k
    cb_pop_front(cothers, 1);
    cb_pop_front(pivot_dy, 1);
    backward_scale_tile();  // cnorm = q
    cb_pop_front(csum, 1);
    broadcast_column_tile(cstatein, 0, cm);  // cm = M in every column
    broadcast_column_tile(cnorm, 0, cs);     // cs = q in every column
    cb_pop_front(cnorm, 1);
    log_softmax_grad_tiles(begin, end);  // cout = dx
    cb_pop_front(cm, 1);
    cb_pop_front(cs, 1);
    cb_pop_front(cstatein, 3);
}
#endif

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0), begin = get_arg_val<uint32_t>(1), end = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(cx, cx, cout);
    reconfig_data_format_srca(cx, cfp32);  // Steps start with source A set for the FP32 statistics.
    cb_wait_front(ccols, 1);
#if defined(CROSS_ENTROPY) && !defined(BACKWARD)
    cb_wait_front(cfirstcol, 1);
#endif
    for (uint32_t row = 0; row < count; ++row) {
#if defined(CROSS_ENTROPY)
        row_statistics(begin, end);  // cmax, cpivot, cnorm = M, K, l
#ifndef BACKWARD
        cross_entropy_loss_tile();  // cout = loss
#else
        cross_entropy_grad_row(begin, end);
#endif
#elif !defined(BACKWARD)
        log_softmax_forward_row(begin, end);
#else
        log_softmax_backward_row(begin, end);
#endif
    }
#if defined(CROSS_ENTROPY) && !defined(BACKWARD)
    cb_pop_front(cfirstcol, 1);
#endif
    cb_pop_front(ccols, 1);
}
