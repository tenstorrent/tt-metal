// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused cross-entropy forward: per row, the loss -((x_t - M) - l). With the vocabulary sharded across devices:
// per-shard records (VP_STATS), and their combination into the loss and the row state (VP_COMBINE). See
// row_softmax_program.hpp. The row statistics and the conventions of the steps are in row_softmax_compute.hpp.
#include "tt-train/sources/ttml/metal/ops/log_softmax/device/kernels/compute/row_softmax_compute.hpp"

constexpr uint32_t chunks = get_compile_time_arg_val(0);
// Circular buffers (see row_softmax_program.cpp), passed to the row-statistics steps as their `Cbs`.
struct Cb {
    static constexpr uint32_t cx = 0, ccols = 2, cm = 3, cs = 4, cic = 5, ci = 6, cmax = 7, cpivot = 10, ctail = 11,
                              cd = 12, cnorm = 13, cout = 16, cpart0 = 17, cpart1 = 18, cpart2 = 19, cpartout = 23,
                              ctargetx = 25, cfirstcol = 27, crecords = 31;
    static constexpr uint32_t cfp32 = ccols;  // Any FP32 buffer, for source-A reconfigurations.
};

#ifdef VP_STATS
// This shard's record of a row: (m, first_v + k, tail without x[k], x_t), each a column vector, for the writer:
// cmax, cpivot, ctail, ctargetx -> crecords. Pops the four.
inline void vocab_parallel_record_tiles(uint32_t first_v_bits) {
    cb_wait_front(Cb::cmax, 1);
    cb_wait_front(Cb::cpivot, 1);
    cb_wait_front(Cb::ctail, 1);
    cb_wait_front(Cb::ctargetx, 1);
    tile_regs_acquire();
    copy_init(Cb::cmax);
    copy_tile(Cb::cmax, 0, 0);  // dst0 = m, this shard's maximum
    copy_init(Cb::cpivot);
    copy_tile(Cb::cpivot, 0, 1);  // dst1 = k, its first-maximum column in the shard
    binop_with_scalar_tile_init();
    add_unary_tile(1, first_v_bits);  // dst1 = first_v + k, the global column
    copy_init(Cb::ctail);
    copy_tile(Cb::ctail, 0, 2);  // dst2 = the shard's tail without x[k]
    copy_init(Cb::ctargetx);
    copy_tile(Cb::ctargetx, 0, 3);  // dst3 = x_t, or 0 when the target is on another shard
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles<Cb::crecords>(4);  // crecords = this shard's record
    tile_regs_release();
    cb_pop_front(Cb::cmax, 1);
    cb_pop_front(Cb::cpivot, 1);
    cb_pop_front(Cb::ctail, 1);
    cb_pop_front(Cb::ctargetx, 1);
}
#endif

#ifdef VP_COMBINE
// The row state (M, l) for the vocabulary-parallel backward: cmax, cnorm -> crecords.
inline void vocab_parallel_row_state_tiles() {
    cb_wait_front(Cb::cmax, 1);
    cb_wait_front(Cb::cnorm, 1);
    tile_regs_acquire();
    copy_init(Cb::cmax);
    copy_tile(Cb::cmax, 0, 0);  // dst0 = M
    copy_init(Cb::cnorm);
    copy_tile(Cb::cnorm, 0, 1);  // dst1 = l
    tile_regs_commit();
    tile_regs_wait();
    pack_n_tiles<Cb::crecords>(2);  // crecords = row state (M, l)
    tile_regs_release();
}
#endif

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0), begin = get_arg_val<uint32_t>(1), end = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t first_v_bits = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup(Cb::cx, Cb::cx, Cb::cout);
    reconfig_data_format_srca(Cb::cx, Cb::cfp32);  // Steps start with source A set for the FP32 statistics.
    cb_wait_front(Cb::ccols, 1);
    cb_wait_front(Cb::cfirstcol, 1);
    for (uint32_t row = 0; row < count; ++row) {
#if defined(VP_STATS)
        row_tail_statistics<Cb, chunks>(begin, end);  // cmax, cpivot, ctail = M, K, T of this shard
        vocab_parallel_record_tiles(first_v_bits);    // crecords = this shard's record
        continue;                                     // A shard's record has no loss.
#elif defined(VP_COMBINE)
        // The shards of a row as lanes: one global first maximum K (lowest shard on ties), the tail without exactly
        // x[K], l = log1p(tail), and the row state (M, l) for the backward.
        combine_lanes<Cb, Cb::cpart0, Cb::cpart1, Cb::cpart2>();  // cmax, cpivot, ctail = M, K, T over every shard
        log_normalizer_tile<Cb>();                                // cnorm = l = log1p(T)
        vocab_parallel_row_state_tiles();                         // crecords = (M, l)
#else
        row_statistics<Cb, chunks>(begin, end);  // cmax, cpivot, cnorm = M, K, l
#endif

        // Loss of the row from its target logit x_t (column 0 of ctargetx; zero for an out-of-range target):
        // -y_t with y_t = (x_t - M) - l, computed exactly as the log_softmax output. When the target is the selected
        // maximum, x_t - M = 0 and the loss is l = log1p(T) itself, however small.
        cb_wait_front(Cb::ctargetx, 1);
        cb_wait_front(Cb::cmax, 1);
        cb_wait_front(Cb::cnorm, 1);
        tile_regs_acquire();
        copy_init(Cb::ctargetx);
        copy_tile(Cb::ctargetx, 0, 0);  // dst0 = x_t
        copy_init(Cb::cmax);
        copy_tile(Cb::cmax, 0, 1);  // dst1 = M
        sfpu_bcast_col_init();
        sfpu_sub_bcast_col(0, 1);  // dst0 = x_t - M
        copy_init(Cb::cnorm);
        copy_tile(Cb::cnorm, 0, 1);  // dst1 = l
        sfpu_bcast_col_init();
        sfpu_sub_bcast_col(0, 1);  // dst0 = y_t = (x_t - M) - l
        negative_tile_init();
        negative_tile(0);  // dst0 = loss = -y_t
        // Keep column 0 only, so the loss tile's padding columns are zero.
        copy_init(Cb::cfirstcol);
        copy_tile(Cb::cfirstcol, 0, 1);  // dst1 = 1 in column 0, 0 elsewhere
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // dst0 = loss in column 0 only
        tile_regs_commit();
        tile_regs_wait();
        pack_one_tile<Cb::cout>(0);  // cout = loss
        tile_regs_release();
        cb_pop_front(Cb::ctargetx, 1);
        cb_pop_front(Cb::cmax, 1);
        cb_pop_front(Cb::cpivot, 1);
        cb_pop_front(Cb::cnorm, 1);
    }
    cb_pop_front(Cb::cfirstcol, 1);
    cb_pop_front(Cb::ccols, 1);
}
