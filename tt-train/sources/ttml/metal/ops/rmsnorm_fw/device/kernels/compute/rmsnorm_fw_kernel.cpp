// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rmsnorm_fw phase B: per (tile-row, slice) work item
//   acc = sum_s partial_s                           (fp32 tile; its row-sums are sum_c x^2 over the whole row)
//   rms = sqrt(rowsum(acc) / C + eps)               (row value broadcast along columns)
//   inv = 1 / rms
//   y   = x * gamma * inv                           (for the item's slice)
// With RETURN_RMS the slice-0 item of each tile-row also emits the rms tile (bf16).

#include "rmsnorm_fw_compute_common.hpp"

void kernel_main() {
    compute_kernel_hw_startup(cb::x, cb::gamma, cb::y);
    cb_wait_front(cb::zero, 1);
    cb_wait_front(cb::ones, 1);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work_start = get_arg_val<uint32_t>(0);
        const SliceGeometry sl = slice_for(work_start + item);

        cb_wait_front(cb::partials, S);

        // acc = sum_s partial_s
        tile_regs_acquire();
        load_tile(cb::partials, 0, 0);
        add_binary_tile_init();
        for (uint32_t s = 1; s < S; ++s) {
            load_tile(cb::partials, s, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        pack_and_push(0, cb::acc);
        cb_pop_front(cb::partials, S);

        // rms = sqrt(rowsum(acc) * (1/C) + eps)
        cb_wait_front(cb::acc, 1);
        tile_regs_acquire();
        matmul_with_constant(cb::acc, cb::ones, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_c_bits);
        add_unary_tile(0, eps_bits);
        sqrt_tile_init();
        sqrt_tile(0);
        tile_regs_commit();
        pack_and_push(0, cb::rms);
        cb_pop_front(cb::acc, 1);
        cb_wait_front(cb::rms, 1);

#ifdef RETURN_RMS
        if (sl.s == 0) {
            tile_regs_acquire();
            load_tile(cb::rms, 0, 0);
            tile_regs_commit();
            pack_and_push(0, cb::rms_out);
        }
#endif

        // inv = 1 / rms
        tile_regs_acquire();
        load_tile(cb::rms, 0, 0);
        recip_tile_init();
        recip_tile(0);
        tile_regs_commit();
        pack_and_push(0, cb::inv);
        cb_pop_front(cb::rms, 1);
        cb_wait_front(cb::inv, 1);

        for (uint32_t c = 0; c < sl.ncols; c += block) {
            const uint32_t n = (c + block <= sl.ncols) ? block : (sl.ncols - c);
            cb_wait_front(cb::x, block);
            cb_wait_front(cb::gamma, block);
            for (uint32_t i = 0; i < n; ++i) {
                // y = (x * gamma) * inv; mul_tiles_bcast_rows accumulates into DEST, so zero it first.
                tile_regs_acquire();
                load_tile(cb::zero, 0, 0);
                reconfig_data_format(cb::x, cb::gamma);
                mul_bcast_rows_init(cb::x, cb::gamma);
                mul_tiles_bcast_rows(cb::x, cb::gamma, i, i, 0);
                load_tile(cb::inv, 0, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                pack_and_push(0, cb::y);
            }
            if (n < block) {
                // Keep the output stream block-aligned; the writer only drains the first n tiles.
                cb_reserve_back(cb::y, block - n);
                cb_push_back(cb::y, block - n);
            }
            cb_pop_front(cb::x, block);
            cb_pop_front(cb::gamma, block);
        }

        cb_pop_front(cb::inv, 1);
    }
}
