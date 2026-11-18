// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused log_softmax: forward y = (x - M) - l with the row state (M, K, l), and the backward from that state
// (BACKWARD). The row statistics and the conventions of the steps are in row_softmax_compute.hpp.
#include "row_softmax_compute.hpp"

constexpr uint32_t chunks = get_compile_time_arg_val(0);
// Circular buffers (see row_softmax_program.cpp), passed to the row-statistics steps as their `Cbs`.
struct Cb {
    static constexpr uint32_t cx = 0, cdy = 1, ccols = 2, cm = 3, cs = 4, cic = 5, ci = 6, cmax = 7, cstateout = 8,
                              cstatein = 9, cpivot = 10, ctail = 11, cd = 12, cnorm = 13, csum = 14, cothers = 15,
                              cout = 16, cpart0 = 17, cpart1 = 18, cpart2 = 19, cpivotdy = 20, cpivotgrad = 21,
                              cpartout = 23;
    static constexpr uint32_t cfp32 = ccols;  // Any FP32 buffer, for source-A reconfigurations.
};

#ifndef BACKWARD
// The row state (M, K, l) for the backward: cmax, cpivot, cnorm -> cstateout.
inline void log_softmax_row_state_tiles() {
    cb_wait_front(Cb::cmax, 1);
    cb_wait_front(Cb::cpivot, 1);
    cb_wait_front(Cb::cnorm, 1);
    tile_regs_acquire();
    copy_init(Cb::cmax);
    copy_tile(Cb::cmax, 0, 0);  // dst0 = M
    copy_init(Cb::cpivot);
    copy_tile(Cb::cpivot, 0, 1);  // dst1 = K
    copy_init(Cb::cnorm);
    copy_tile(Cb::cnorm, 0, 2);  // dst2 = l
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles<Cb::cstateout>(3);  // cstateout = row state (M, K, l) for the backward
    tile_regs_release();
}

// Pass 2, native-precision output y = (x - M) - l over this unit's logits, with M (cm) and l (cs) broadcast across
// the row: cx -> cout. Pops each logits tile.
inline void log_softmax_output_tiles(uint32_t begin, uint32_t end) {
    cb_wait_front(Cb::cm, 1);
    cb_wait_front(Cb::cs, 1);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(Cb::cx, 1);
        tile_regs_acquire();
        copy_init(Cb::cm);
        copy_tile(Cb::cm, 0, 1);  // dst1 = M
        copy_init(Cb::cs);
        copy_tile(Cb::cs, 0, 2);  // dst2 = l
        reconfig_data_format_srca(Cb::cfp32, Cb::cx);
        copy_init(Cb::cx);
        copy_tile(Cb::cx, 0, 0);  // dst0 = x
        reconfig_data_format_srca(Cb::cx, Cb::cfp32);
        log_softmax_output(0);  // dst0 = y = (x - M) - l
        tile_regs_commit();
        tile_regs_wait();
        pack_one_tile<Cb::cout>(0);  // cout = y
        tile_regs_release();
        cb_pop_front(Cb::cx, 1);
    }
}

// One row of the log_softmax forward: its statistics, the row state for the backward, then the output.
inline void log_softmax_forward_row(uint32_t begin, uint32_t end) {
    row_statistics<Cb, chunks>(begin, end);  // cmax, cpivot, cnorm = M, K, l
    log_softmax_row_state_tiles();           // cstateout = (M, K, l)
    cb_pop_front(Cb::cpivot, 1);
    broadcast_column_tile<Cb::cmax, Cb::cm>(0);   // cm = M in every column
    broadcast_column_tile<Cb::cnorm, Cb::cs>(0);  // cs = l in every column
    cb_pop_front(Cb::cmax, 1);
    cb_pop_front(Cb::cnorm, 1);
    log_softmax_output_tiles(begin, end);  // cout = y
    cb_pop_front(Cb::cm, 1);
    cb_pop_front(Cb::cs, 1);
}
#endif

#ifdef BACKWARD
// Pass 1 of the backward, per lane and then per row: A, the sum of the upstream gradient over this unit's columns
// except the selected element (the reader zeroes it and hands it over separately in cpivotdy): cdy -> cothers. Pops
// each gradient tile.
inline void upstream_sum_tiles(uint32_t begin, uint32_t end) {
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.F);  // dst0 = 0: the lanes' running sum of dy
    reconfig_data_format_srca(Cb::cfp32, Cb::cdy);
    copy_init(Cb::cdy);
    add_binary_tile_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(Cb::cdy, 1);
        copy_tile(Cb::cdy, 0, 1);  // dst1 = dy
        add_binary_tile(0, 1, 0);  // dst0 += dy
        cb_pop_front(Cb::cdy, 1);
    }
    reconfig_data_format_srca(Cb::cdy, Cb::cfp32);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = A: sum of dy without dy_k
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cb::cothers>(0);  // cothers = A
    tile_regs_release();
}

// This core's (A, dy_k), published for the other cores of the row: cothers, cpivotdy -> cpartout. Pops the two.
inline void partial_upstream_sums_tiles() {
    cb_wait_front(Cb::cothers, 1);
    cb_wait_front(Cb::cpivotdy, 1);
    tile_regs_acquire();
    copy_init(Cb::cothers);
    copy_tile(Cb::cothers, 0, 0);  // dst0 = A
    copy_init(Cb::cpivotdy);
    copy_tile(Cb::cpivotdy, 0, 1);  // dst1 = dy_k, or 0 when another core holds k
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles<Cb::cpartout>(2);  // cpartout = this core's (A, dy_k)
    tile_regs_release();
    cb_pop_front(Cb::cothers, 1);
    cb_pop_front(Cb::cpivotdy, 1);
}

// The sum over lanes (the cores of the row) of each row of `cb_in` -> `cb_out`. Pops cb_in.
template <uint32_t cb_in, uint32_t cb_out>
inline void sum_over_lanes_tile() {
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_init(cb_in);
    copy_tile(cb_in, 0, 0);  // dst0 = one value per core, one per lane
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);  // dst0 = their sum
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<cb_out>(0);  // cb_out = the sum
    tile_regs_release();
    cb_pop_front(cb_in, 1);
}

// S = A + dy_k, the row's full sum of the upstream gradient: cothers, pivot_dy -> csum.
template <uint32_t pivot_dy>
inline void upstream_total_tile() {
    cb_wait_front(Cb::cothers, 1);
    cb_wait_front(pivot_dy, 1);
    tile_regs_acquire();
    copy_init(Cb::cothers);
    copy_tile(Cb::cothers, 0, 0);  // dst0 = A
    copy_init(pivot_dy);
    copy_tile(pivot_dy, 0, 1);  // dst1 = dy_k
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dst0 = S = A + dy_k
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cb::csum>(0);  // csum = S
    tile_regs_release();
}

// Eq. (7) in closed form, the selected element's gradient dx_k = -sum_{j!=k} (dy_j - p_j S) = S * q' - A with
// q' = sum_{j!=k} p_j = T / (1 + T) = -expm1(-log1p(T)), accurate when p_k rounds to one:
// cstatein (l), csum, cothers -> cpivotgrad, which the writer substitutes.
inline void selected_gradient_tile() {
    cb_wait_front(Cb::csum, 1);
    tile_regs_acquire();
    copy_init(Cb::cstatein);
    copy_tile(Cb::cstatein, 2, 0);  // dst0 = l
    negative_tile_init();
    negative_tile(0);  // dst0 = -l
    expm1_tile_init();
    expm1_tile(0);  // dst0 = expm1(-l) = -T / (1 + T)
    copy_init(Cb::csum);
    copy_tile(Cb::csum, 0, 1);  // dst1 = S
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);  // dst0 = -S * T / (1 + T)
    copy_init(Cb::cothers);
    copy_tile(Cb::cothers, 0, 1);  // dst1 = A
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dst0 = A - S * T / (1 + T)
    negative_tile_init();
    negative_tile(0);  // dst0 = dx_k = S * T / (1 + T) - A
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cb::cpivotgrad>(0);  // cpivotgrad = dx_k
    tile_regs_release();
}

// q = S * exp(-l) = S * p_k, so that softmax(x) * S = exp(x - M) * q: cstatein (l), csum -> cnorm.
inline void backward_scale_tile() {
    tile_regs_acquire();
    copy_init(Cb::cstatein);
    copy_tile(Cb::cstatein, 2, 0);  // dst0 = l
    negative_tile_init();
    negative_tile(0);  // dst0 = -l
    exp_tile_init<USE_APPROXIMATE_EXPONENTIAL>();
    exp_tile<USE_APPROXIMATE_EXPONENTIAL>(0);  // dst0 = p_k = exp(-l)
    copy_init(Cb::csum);
    copy_tile(Cb::csum, 0, 1);  // dst1 = S
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);  // dst0 = q = S * p_k
    tile_regs_commit();
    tile_regs_wait();
    pack_one_tile<Cb::cnorm>(0);  // cnorm = q
    tile_regs_release();
}

// Pass 2, dx = dy - exp(x - M) * q over this unit's columns, with M (cm) and q (cs) broadcast across the row:
// cx, cdy -> cout. Pops each logits and gradient tile. The writer substitutes dx_k.
inline void log_softmax_grad_tiles(uint32_t begin, uint32_t end) {
    cb_wait_front(Cb::cm, 1);
    cb_wait_front(Cb::cs, 1);
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(Cb::cx, 1);
        cb_wait_front(Cb::cdy, 1);
        tile_regs_acquire();
        copy_init(Cb::cm);
        copy_tile(Cb::cm, 0, 2);  // dst2 = M
        copy_init(Cb::cs);
        copy_tile(Cb::cs, 0, 3);  // dst3 = q
        reconfig_data_format_srca(Cb::cfp32, Cb::cx);
        copy_init(Cb::cx);
        copy_tile(Cb::cx, 0, 0);  // dst0 = x
        reconfig_data_format_srca(Cb::cx, Cb::cdy);
        copy_init(Cb::cdy);
        copy_tile(Cb::cdy, 0, 1);  // dst1 = dy
        reconfig_data_format_srca(Cb::cdy, Cb::cfp32);
        log_softmax_grad(0);  // dst0 = dx = dy - exp(x - M) * q
        tile_regs_commit();
        tile_regs_wait();
        pack_one_tile<Cb::cout>(0);  // cout = dx
        tile_regs_release();
        cb_pop_front(Cb::cx, 1);
        cb_pop_front(Cb::cdy, 1);
    }
}

// One row of the log_softmax backward, from the row state (M, K, l) in cstatein.
inline void log_softmax_backward_row(uint32_t begin, uint32_t end) {
    cb_wait_front(Cb::cstatein, 3);
    upstream_sum_tiles(begin, end);  // cothers = A
    // dy_k: this core's own (cpivotdy), or the row's, summed over the cores sharing it (cd).
    constexpr uint32_t pivot_dy = chunks > 1 ? Cb::cd : Cb::cpivotdy;
    if constexpr (chunks > 1) {
        partial_upstream_sums_tiles();                   // cpartout = this core's (A, dy_k)
        sum_over_lanes_tile<Cb::cpart0, Cb::cothers>();  // cothers = A of the whole row
        sum_over_lanes_tile<Cb::cpart1, Cb::cd>();       // cd = dy_k
    }
    upstream_total_tile<pivot_dy>();  // csum = S
    selected_gradient_tile();         // cpivotgrad = dx_k
    cb_pop_front(Cb::cothers, 1);
    cb_pop_front(pivot_dy, 1);
    backward_scale_tile();  // cnorm = q
    cb_pop_front(Cb::csum, 1);
    broadcast_column_tile<Cb::cstatein, Cb::cm>(0);  // cm = M in every column
    broadcast_column_tile<Cb::cnorm, Cb::cs>(0);     // cs = q in every column
    cb_pop_front(Cb::cnorm, 1);
    log_softmax_grad_tiles(begin, end);  // cout = dx
    cb_pop_front(Cb::cm, 1);
    cb_pop_front(Cb::cs, 1);
    cb_pop_front(Cb::cstatein, 3);
}
#endif

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0), begin = get_arg_val<uint32_t>(1), end = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(Cb::cx, Cb::cx, Cb::cout);
    reconfig_data_format_srca(Cb::cx, Cb::cfp32);  // Steps start with source A set for the FP32 statistics.
    cb_wait_front(Cb::ccols, 1);
    for (uint32_t row = 0; row < count; ++row) {
#ifndef BACKWARD
        log_softmax_forward_row(begin, end);
#else
        log_softmax_backward_row(begin, end);
#endif
    }
    cb_pop_front(Cb::ccols, 1);
}
