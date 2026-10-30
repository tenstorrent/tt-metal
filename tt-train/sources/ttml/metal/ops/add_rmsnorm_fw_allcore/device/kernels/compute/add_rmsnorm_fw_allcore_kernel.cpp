// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// add_rmsnorm_fw phase B: per (tile-row, slice) work item
//   acc = sum_s partial_s                      (elementwise over the row's S partial tiles)
//   rms = sqrt(rowsum(acc) / C + eps)          (row value broadcast along columns; packed to the rms output too)
//   inv = 1 / rms
//   out = sum * gamma * inv

#include "add_rmsnorm_fw_allcore_compute_common.hpp"

void kernel_main() {
    compute_kernel_hw_startup(cb::sum, cb::gamma, cb::out);
    cb_wait_front(cb::zero, 1);
    cb_wait_front(cb::ones, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work_start = get_arg_val<uint32_t>(0);
        const SliceGeometry sl = slice_for(work_start + item);

        // acc = sum_s partial_s
        cb_wait_front(cb::partials, S);
        tile_regs_acquire();
        load_tile(cb::partials, 0, 0);
        add_binary_tile_init();
        for (uint32_t s = 1; s < S; ++s) {
            load_tile(cb::partials, s, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        pack_and_push(0, cb::row_acc);
        cb_pop_front(cb::partials, S);

        // rms = sqrt(rowsum(acc) / C + eps), broadcast along columns
        cb_wait_front(cb::row_acc, 1);
        tile_regs_acquire();
        matmul_with_constant(cb::row_acc, cb::ones, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_c_bits);
        add_unary_tile(0, eps_bits);
        sqrt_tile_init();
        sqrt_tile(0);
        tile_regs_commit();
        cb_reserve_back(cb::rms_tmp, 1);
        cb_reserve_back(cb::rms_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb::rms_tmp);
        pack_tile(0, cb::rms_tmp);
        pack_reconfig_data_format(cb::rms_out);
        pack_tile(0, cb::rms_out);
        tile_regs_release();
        cb_push_back(cb::rms_tmp, 1);
        cb_push_back(cb::rms_out, 1);
        cb_pop_front(cb::row_acc, 1);

        // inv = 1 / rms
        cb_wait_front(cb::rms_tmp, 1);
        tile_regs_acquire();
        load_tile(cb::rms_tmp, 0, 0);
        recip_tile_init();
        recip_tile(0);
        tile_regs_commit();
        pack_and_push(0, cb::inv);
        cb_pop_front(cb::rms_tmp, 1);
        cb_wait_front(cb::inv, 1);

        for (uint32_t c = 0; c < sl.ncols; c += block) {
            const uint32_t n = (c + block <= sl.ncols) ? block : (sl.ncols - c);
            // Full-block CB traffic (see the reader): wait and pop `block`, use the first n tiles, pad the output
            // of a shorter tail to a full block.
            cb_wait_front(cb::sum, block);
            cb_wait_front(cb::gamma, block);
            for (uint32_t i = 0; i < n; ++i) {
                // out = (sum * gamma) * inv; the row-broadcast multiply accumulates into DEST, so zero it first.
                tile_regs_acquire();
                load_tile(cb::zero, 0, 0);
                reconfig_data_format(cb::sum, cb::gamma);
                mul_bcast_rows_init(cb::sum, cb::gamma);
                mul_tiles_bcast_rows(cb::sum, cb::gamma, i, i, 0);
                load_tile(cb::inv, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                pack_and_push(0, cb::out);
            }
            for (uint32_t i = n; i < block; ++i) {  // tail padding (unused tiles)
                tile_regs_acquire();
                tile_regs_commit();
                pack_and_push(0, cb::out);
            }
            cb_pop_front(cb::sum, block);
            cb_pop_front(cb::gamma, block);
        }

        cb_pop_front(cb::inv, 1);
    }
}
