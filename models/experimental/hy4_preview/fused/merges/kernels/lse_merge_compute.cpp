// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-chip log-sum-exp merge (fused/merges/lse_merge.py): the compute kernel of a merge core. A chain of
// pairwise merges of partials (O, max, sum) of one user and one head group with the flash-attention tail LLK
// (api/compute/experimental/sdpa.h sdpa_tail, as fused/sparse_mla merges its S-blocks):
//
//     m = max(m1, m2);  P1 = exp(scale (m1 - m));  P2 = exp(scale (m2 - m));  s = s1 P1 + s2 P2;  O = O1 P1 + O2 P2
//
//   column stage   the running partial (cb_acc) with every other slot of the column landing, slots ascending
//   row stage      then with every other slot of the row landing
//   sink           then with the sink peer, and the result divided by the sum: ctx [8 heads, 512] -> cb_final
// A landing slot is [max / sum tile | 16 O tiles]: the same circular buffer is the LLK's max / sum input and its
// O input. The slot of this chip itself is popped unused (the column landing's is never written, the row
// landing's is this chip's own copy).
//
// Compile-time args: column members, row members, cb_acc_ms, cb_acc_o, cb_send_ms, cb_send_o, cb_sink, cb_final,
// cb_col0, cb_col1, cb_row0, cb_row1, scale (fp32 bits).
// Runtime args: row slot, column slot.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/experimental/sdpa.h"

#include "collective_compute.hpp"
#include "lse_common.hpp"

using namespace hy4_lse;

constexpr uint32_t col_members = get_compile_time_arg_val(0);
constexpr uint32_t row_members = get_compile_time_arg_val(1);
constexpr uint32_t cb_acc_ms = get_compile_time_arg_val(2);
constexpr uint32_t cb_acc_o = get_compile_time_arg_val(3);
constexpr uint32_t cb_send_ms = get_compile_time_arg_val(4);
constexpr uint32_t cb_send_o = get_compile_time_arg_val(5);
constexpr uint32_t cb_sink = get_compile_time_arg_val(6);
constexpr uint32_t cb_final = get_compile_time_arg_val(7);
constexpr uint32_t cb_col0 = get_compile_time_arg_val(8);
constexpr uint32_t cb_col1 = get_compile_time_arg_val(9);
constexpr uint32_t cb_row0 = get_compile_time_arg_val(10);
constexpr uint32_t cb_row1 = get_compile_time_arg_val(11);
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(12);
constexpr bool exp_approx_mode = false;
constexpr uint32_t num_blocks = 2;
constexpr uint32_t block_size = O_TILES / num_blocks;
static_assert(cb_acc_o == 0, "the hardware startup is seeded from circular buffer 0");

namespace {
// worker (a landing or the sink) merged into prev; the result goes to cur.
template <bool normalize>
ALWI void merge(uint32_t worker, uint32_t prev_ms, uint32_t prev_o, uint32_t cur_ms, uint32_t cur_o) {
    sdpa_tail<exp_approx_mode, normalize, block_size, num_blocks, scale_fp32, VectorMode::C>(
        worker, prev_ms, cur_ms, worker, prev_o, cur_o);
}

ALWI void skip(uint32_t cb) {
    cb_wait_front(cb, PART_TILES);
    cb_pop_front(cb, PART_TILES);
}
}  // namespace

void kernel_main() {
#ifdef HY4_MERGE_NOP
    return;
#else
    const uint32_t row = get_arg_val<uint32_t>(0);
    const uint32_t col = get_arg_val<uint32_t>(1);

    deepseek_compute_kernel_init();
    reconfig_full_operand_srca(cb_sink);
    exp_tile_init<exp_approx_mode, scale_fp32>();

    if constexpr (col_members > 1) {
        const uint32_t landing = hy4_collective::wait_landing(cb_col0, cb_col1, col_members * PART_TILES);
        uint32_t done = 0;
        for (uint32_t slot = 0; slot < col_members; ++slot) {
            if (slot == row) {
                skip(landing);
                continue;
            }
            const bool to_send = row_members > 1 && ++done == col_members - 1;
            merge<false>(
                landing, cb_acc_ms, cb_acc_o, to_send ? cb_send_ms : cb_acc_ms, to_send ? cb_send_o : cb_acc_o);
        }
    }
    if constexpr (row_members > 1) {
        const uint32_t landing = hy4_collective::wait_landing(cb_row0, cb_row1, row_members * PART_TILES);
        bool first = true;
        for (uint32_t slot = 0; slot < row_members; ++slot) {
            if (slot == col) {
                skip(landing);
                continue;
            }
            merge<false>(landing, first ? cb_send_ms : cb_acc_ms, first ? cb_send_o : cb_acc_o, cb_acc_ms, cb_acc_o);
            first = false;
        }
    }
    merge<true>(cb_sink, cb_acc_ms, cb_acc_o, cb_acc_ms, cb_final);
#endif
}
