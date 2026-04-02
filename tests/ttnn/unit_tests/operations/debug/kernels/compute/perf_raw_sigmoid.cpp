// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf baseline: raw LLK sigmoid (single op, different from exp)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    init_sfpu(cb_in, cb_out);

    for (uint32_t block = 0; block < per_core_block_cnt; block++) {
        for (uint32_t tile = 0; tile < per_core_block_dim; tile++) {
            cb_wait_front(cb_in, 1);
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();

            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, 0, 0);

            sigmoid_tile_init();
            sigmoid_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_pop_front(cb_in, 1);
            cb_push_back(cb_out, 1);
        }
    }
}
