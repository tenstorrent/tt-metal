// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Online-softmax (logsumexp) merge of one ring step's partial attention result, per tile row of
// 32 sequence positions:
//
//   m       = max(running, lse_i)
//   new_lse = m + log(exp(running - m) + exp(lse_i - m))
//   w_old   = exp(running - new_lse)
//   w_new   = exp(lse_i   - new_lse)
//   accum   = accum * w_old + float(O_i) * w_new        (w_* broadcast along the row)
//
// This is the same sequence of operations, in the same order, as the ttnn composite it replaces,
// all in FP32 on the SFPU (DST is FP32, FP32 CBs are unpacked straight to DST, and the row weights
// are column-broadcast through the unpack-to-dest path, never through the 19-bit SrcB).
//
// -inf semantics (exp(0) == 1, exp(-inf) == +0 and log(1) == 0 exactly on the FP32 SFPU path):
//   lse_i = -inf, running finite: new_lse = running, w_old = 1, w_new = 0 -> accum unchanged.
//   running = -inf, lse_i finite: new_lse = lse_i, w_old = 0, w_new = 1 -> accum = O_i.
//   both -inf: NaN, as in the composite (cannot happen in ring attention: step 0 always contributes).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/mask.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);

// Inputs (reader)
constexpr uint32_t cb_running_lse = tt::CBIndex::c_0;  // FP32, unpack to DST
constexpr uint32_t cb_step_lse = tt::CBIndex::c_1;     // FP32, unpack to DST
constexpr uint32_t cb_accum = tt::CBIndex::c_2;        // FP32, unpack to DST
constexpr uint32_t cb_step_out = tt::CBIndex::c_3;     // BF16, through SrcA (exact in FP32 DST)
constexpr uint32_t cb_col0_mask = tt::CBIndex::c_4;    // BF16, 1.0 in column 0, 0.0 elsewhere
// Outputs (writer)
constexpr uint32_t cb_new_lse = tt::CBIndex::c_5;  // FP32
constexpr uint32_t cb_out = tt::CBIndex::c_9;      // FP32
// Compute-internal
constexpr uint32_t cb_new_lse_scratch = tt::CBIndex::c_6;  // FP32, unpack to DST
constexpr uint32_t cb_w_old = tt::CBIndex::c_7;            // FP32, unpack to DST (column broadcast)
constexpr uint32_t cb_w_new = tt::CBIndex::c_8;            // FP32, unpack to DST (column broadcast)

// new_lse = logaddexp(running, lse_i), masked to column 0 so that the running logsumexp keeps the
// SDPA intermediates layout (zeros in columns 1..31). Packed twice: once for the writer, once for
// the weight pass below (the writer pops its copy whenever it is done with it).
// SrcA is left configured for cb_col0_mask.
inline void compute_new_lse() {
    constexpr uint32_t reg_a = 0U;  // running, then exp(running - m), then the sum, then new_lse
    constexpr uint32_t reg_b = 1U;  // lse_i, then exp(lse_i - m), then the column-0 mask
    constexpr uint32_t reg_m = 2U;  // m

    cb_wait_front(cb_running_lse, onetile);
    cb_wait_front(cb_step_lse, onetile);

    tile_regs_acquire();
    reconfig_data_format_srca(cb_col0_mask, cb_running_lse);
    copy_init(cb_running_lse);
    copy_tile(cb_running_lse, /* tile_idx */ 0, reg_a);
    copy_init(cb_step_lse);
    copy_tile(cb_step_lse, /* tile_idx */ 0, reg_b);

    binary_max_tile_init();
    binary_max_tile(reg_a, reg_b, reg_m);

    sub_binary_tile_init();
    sub_binary_tile(reg_a, reg_m, reg_a);
    sub_binary_tile(reg_b, reg_m, reg_b);

    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(reg_a);
    exp_tile</* approx */ false>(reg_b);

    add_binary_tile_init();
    add_binary_tile(reg_a, reg_b, reg_a);

    log_tile_init</* fast_and_approx */ false>();
    log_tile</* fast_and_approx */ false>(reg_a);

    add_binary_tile_init();
    add_binary_tile(reg_m, reg_a, reg_a);

    // mask_tile needs the mask in the register right after the data.
    reconfig_data_format_srca(cb_step_lse, cb_col0_mask);
    copy_init(cb_col0_mask);
    copy_tile(cb_col0_mask, /* tile_idx */ 0, reg_b);
    mask_tile_init();
    mask_tile(reg_a, reg_b);
    tile_regs_commit();

    pack_and_push_two_tiles(reg_a, cb_new_lse, reg_a, cb_new_lse_scratch);
}

// w_old = exp(running - new_lse), w_new = exp(lse_i - new_lse). Only column 0 is used (broadcast
// along the row below). Consumes the row's running and step logsumexp tiles and the scratch copy.
// SrcA is left configured for FP32 (cb_step_lse).
inline void compute_weights() {
    constexpr uint32_t reg_old = 0U;
    constexpr uint32_t reg_lse = 1U;
    constexpr uint32_t reg_new = 2U;

    cb_wait_front(cb_new_lse_scratch, onetile);

    tile_regs_acquire();
    reconfig_data_format_srca(cb_col0_mask, cb_running_lse);
    copy_init(cb_running_lse);
    copy_tile(cb_running_lse, /* tile_idx */ 0, reg_old);
    copy_init(cb_new_lse_scratch);
    copy_tile(cb_new_lse_scratch, /* tile_idx */ 0, reg_lse);
    copy_init(cb_step_lse);
    copy_tile(cb_step_lse, /* tile_idx */ 0, reg_new);

    sub_binary_tile_init();
    sub_binary_tile(reg_old, reg_lse, reg_old);
    sub_binary_tile(reg_new, reg_lse, reg_new);

    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(reg_old);
    exp_tile</* approx */ false>(reg_new);
    tile_regs_commit();

    pack_and_push_two_tiles(reg_old, cb_w_old, reg_new, cb_w_new);

    cb_pop_front(cb_running_lse, onetile);
    cb_pop_front(cb_step_lse, onetile);
    cb_pop_front(cb_new_lse_scratch, onetile);
}

// accum = accum * w_old + float(O_i) * w_new for one tile. SrcA enters and leaves configured for
// FP32 (the FP32 CBs all share one format, so only the bf16 step output needs a reconfig).
inline void merge_tile(const uint32_t tile_idx) {
    constexpr uint32_t reg_accum = 0U;
    constexpr uint32_t reg_w_old = 1U;
    constexpr uint32_t reg_step = 2U;
    constexpr uint32_t reg_w_new = 3U;

    tile_regs_acquire();
    copy_init(cb_accum);
    copy_tile(cb_accum, tile_idx, reg_accum);

    unary_bcast_init<BroadcastType::COL>(cb_w_old);
    unary_bcast<BroadcastType::COL>(cb_w_old, /* tile_idx */ 0, reg_w_old);
    unary_bcast<BroadcastType::COL>(cb_w_new, /* tile_idx */ 0, reg_w_new);
    unary_bcast_uninit<BroadcastType::COL>(cb_w_new);

    reconfig_data_format_srca(cb_accum, cb_step_out);
    copy_init(cb_step_out);
    copy_tile(cb_step_out, tile_idx, reg_step);
    reconfig_data_format_srca(cb_step_out, cb_accum);

    mul_binary_tile_init();
    mul_binary_tile(reg_accum, reg_w_old, reg_accum);
    mul_binary_tile(reg_step, reg_w_new, reg_step);

    add_binary_tile_init();
    add_binary_tile(reg_accum, reg_step, reg_accum);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(reg_accum, cb_out);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    cb_wait_front(cb_col0_mask, onetile);

    compute_kernel_hw_startup(cb_col0_mask, cb_col0_mask, cb_out);
    // Every packed tile (new_lse, weights, accumulator) is FP32: one packer configuration for the whole kernel.
    pack_reconfig_data_format(cb_out);

    for (uint32_t row = 0; row < num_rows; ++row) {
        compute_new_lse();
        compute_weights();

        cb_wait_front(cb_w_old, onetile);
        cb_wait_front(cb_w_new, onetile);
        for (uint32_t col = 0; col < Wt; col += block_size) {
            cb_wait_front(cb_accum, block_size);
            cb_wait_front(cb_step_out, block_size);
            cb_reserve_back(cb_out, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                merge_tile(block_idx);
            }
            cb_push_back(cb_out, block_size);
            cb_pop_front(cb_accum, block_size);
            cb_pop_front(cb_step_out, block_size);
        }
        cb_pop_front(cb_w_old, onetile);
        cb_pop_front(cb_w_new, onetile);

        // compute_new_lse starts from the mask's SrcA format.
        reconfig_data_format_srca(cb_accum, cb_col0_mask);
    }
}
