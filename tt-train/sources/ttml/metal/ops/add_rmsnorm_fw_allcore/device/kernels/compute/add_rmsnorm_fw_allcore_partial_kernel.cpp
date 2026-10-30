// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rmsnorm_fw phase A: per (tile-row, slice) work item
//   sum_j   = residual_j + delta_j                  (packed to bf16, the next residual stream)
//   partial = sum_{tiles j in slice} sum_j^2        (fp32, from the fp32 sum before its bf16 rounding)
// DEST: register 0 holds the running partial across the item's blocks (round-tripped through cb::acc between
// blocks, since DEST does not persist across a pack), registers 1..block the block's sums, block+1 the scratch.

#include "add_rmsnorm_fw_allcore_compute_common.hpp"

void kernel_main() {
    compute_kernel_hw_startup(cb::residual, cb::delta, cb::sum_out);
    cb_wait_front(cb::zero, 1);
    constexpr uint32_t scratch = block + 1U;

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work_start = get_arg_val<uint32_t>(0);
        const SliceGeometry sl = slice_for(work_start + item);

        for (uint32_t c = 0; c < sl.ncols; c += block) {
            const uint32_t n = (c + block <= sl.ncols) ? block : (sl.ncols - c);
            const bool last = c + n >= sl.ncols;
            // Full-block CB traffic (see the reader): wait and pop `block`, use the first n tiles, pad the sum
            // output of a shorter tail to a full block.
            cb_wait_front(cb::residual, block);
            cb_wait_front(cb::delta, block);

            tile_regs_acquire();
            if (c == 0) {
                load_tile(cb::zero, 0, 0);
            } else {
                cb_wait_front(cb::acc, 1);
                load_tile(cb::acc, 0, 0);
                cb_pop_front(cb::acc, 1);
            }
            for (uint32_t i = 0; i < n; ++i) {
                load_tile(cb::residual, i, 1U + i);
                load_tile(cb::delta, i, scratch);
                add_binary_tile_init();
                add_binary_tile(1U + i, scratch, 1U + i);
                mul_binary_tile_init();
                mul_binary_tile(1U + i, 1U + i, scratch);
                add_binary_tile_init();
                add_binary_tile(0, scratch, 0);
            }
            tile_regs_commit();

            const uint32_t cb_acc_out = last ? cb::partial_out : cb::acc;
            cb_reserve_back(cb::sum_out, block);
            cb_reserve_back(cb_acc_out, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb::sum_out);
            for (uint32_t i = 0; i < block; ++i) {
                pack_tile(i < n ? 1U + i : 1U, cb::sum_out);  // tail padding repeats a tile (unused)
            }
            pack_reconfig_data_format(cb_acc_out);
            pack_tile(0, cb_acc_out);
            tile_regs_release();
            cb_push_back(cb::sum_out, block);
            cb_push_back(cb_acc_out, 1);

            cb_pop_front(cb::residual, block);
            cb_pop_front(cb::delta, block);
        }
    }
}
