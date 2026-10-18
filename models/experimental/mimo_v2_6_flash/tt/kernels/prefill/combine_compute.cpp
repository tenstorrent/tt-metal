// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISCs of the prefill MoE combine. Per item: the K row-major blocks (32 rows x block tile columns, as block
// tile-sized pages, all K in cb_in) summed in fp32 dest 4 pages at a time (an elementwise sum does not care that the
// pages hold rows, not tiles; dest 0-3 accumulate, 4-7 take the next k), packed to cb_sum, then tilized into the
// item's block output tiles.
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
    constexpr uint32_t block = get_compile_time_arg_val(4);
    constexpr uint32_t chunk = 4;
    const uint32_t count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_sum);
    for (uint32_t i = 0; i < count; i++) {
        cb_wait_front(cb_in, K * block);
        cb_reserve_back(cb_sum, block);
        for (uint32_t c0 = 0; c0 < block; c0 += chunk) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < K; k++) {
                copy_init(cb_in);
                for (uint32_t c = 0; c < chunk; c++) {
                    copy_tile(cb_in, k * block + c0 + c, k == 0 ? c : chunk + c);
                }
                if (k > 0) {
                    add_binary_tile_init();
                    for (uint32_t c = 0; c < chunk; c++) {
                        add_binary_tile(c, chunk + c, c);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t c = 0; c < chunk; c++) {
                pack_tile<true>(c, cb_sum, c0 + c);
            }
            tile_regs_release();
        }
        cb_push_back(cb_sum, block);
        cb_pop_front(cb_in, K * block);

        tilize_init(cb_sum, block, cb_out);
        cb_wait_front(cb_sum, block);
        cb_reserve_back(cb_out, block);
        tilize_block(cb_sum, block, cb_out);
        cb_push_back(cb_out, block);
        cb_pop_front(cb_sum, block);
        tilize_uninit(cb_sum, cb_out);
    }
}
