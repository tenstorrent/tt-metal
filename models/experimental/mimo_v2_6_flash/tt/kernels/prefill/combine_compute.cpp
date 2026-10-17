// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISCs of the prefill MoE combine. Per item: the K row-major blocks (32 rows x 4 tile columns, as 4 tile-sized
// pages) summed in fp32 dest page by page (an elementwise sum does not care that the pages hold rows, not tiles;
// dest 0-3 accumulate, 4-7 take the next k), packed to cb_sum, then tilized into the item's 4 output tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tilize.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t block = 4;
    const uint32_t count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_sum);
    for (uint32_t i = 0; i < count; i++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < K; k++) {
            cb_wait_front(cb_in, block);
            copy_init(cb_in);
            for (uint32_t c = 0; c < block; c++) {
                copy_tile(cb_in, c, k == 0 ? c : block + c);
            }
            if (k > 0) {
                add_binary_tile_init();
                for (uint32_t c = 0; c < block; c++) {
                    add_binary_tile(c, block + c, c);
                }
            }
            cb_pop_front(cb_in, block);
        }
        tile_regs_commit();
        cb_reserve_back(cb_sum, block);
        tile_regs_wait();
        for (uint32_t c = 0; c < block; c++) {
            pack_tile(c, cb_sum);
        }
        tile_regs_release();
        cb_push_back(cb_sum, block);

        tilize_init(cb_sum, block, cb_out);
        cb_wait_front(cb_sum, block);
        cb_reserve_back(cb_out, block);
        tilize_block(cb_sum, block, cb_out);
        cb_push_back(cb_out, block);
        cb_pop_front(cb_sum, block);
        tilize_uninit(cb_sum, cb_out);
    }
}
