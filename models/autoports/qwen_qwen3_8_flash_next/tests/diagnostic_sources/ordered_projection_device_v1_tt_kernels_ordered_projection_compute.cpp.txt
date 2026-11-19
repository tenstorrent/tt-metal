// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t cb_left = 2, cb_right = 3, cb_out = 4;
    compute_kernel_hw_startup(cb_left, cb_right, cb_out);
    tile_regs_acquire();
    // Tile2 is partial0, then the total; tile3 holds independently zeroed
    // later partials. No partial is packed/reloaded or routed through TF32 SrcA.
    const uint32_t block_width = get_arg_val<uint32_t>(0);
    const uint32_t blocks = (2560 + block_width - 1) / block_width;
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t accumulator = block == 0 ? 2 : 3;
        fill_tile_init();
        fill_tile(accumulator, 0.0f);
        const uint32_t remaining = 2560 - block * block_width;
        const uint32_t width = remaining < block_width ? remaining : block_width;
        for (uint32_t k = 0; k < width; ++k) {
            cb_wait_front(cb_left, 1);
            cb_wait_front(cb_right, 1);
            reconfig_data_format_srca(cb_left);
            copy_init(cb_left);
            copy_tile(cb_left, 0, 0);
            reconfig_data_format_srca(cb_right);
            copy_init(cb_right);
            copy_tile(cb_right, 0, 1);
            // BF16 products are exact FP32 for the normal finite target
            // operands. Keep each product and accumulator addition separate:
            // the reader sends high,low; there is no pair sum or tree reduce.
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            add_binary_tile_init();
            add_binary_tile(accumulator, 0, accumulator);
            cb_pop_front(cb_left, 1);
            cb_pop_front(cb_right, 1);
        }
        if (block != 0) {
            // Pinned brgemm: each new partial + accumulated earlier partials.
            add_binary_tile_init();
            add_binary_tile(3, 2, 2);
        }
    }
    cb_reserve_back(cb_out, 1);
    tile_regs_commit();
    tile_regs_wait();
    // One final BF16 boundary. fp32_dest_acc_en retains the partial mantissas;
    // normal non-stochastic pack RNE must be checked by the exact-output probe.
    pack_tile(2, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
