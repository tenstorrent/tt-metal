// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode all-reduce, step 3 compute (Laguna): DST 0 = sum of the D + 1 flat element blocks (the D partials,
// then the residual) in fp32 DEST; packed bf16. The add is elementwise, so the blocks' row-major order survives.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t D = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = 0, cb_out = 16;
    compute_kernel_hw_startup(cb_x, cb_out);
    cb_wait_front(cb_x, D + 1);
    tile_regs_acquire();
    copy_tile_init(cb_x);
    copy_tile(cb_x, 0, 0);
    for (uint32_t i = 1; i <= D; ++i) {
        copy_tile_init(cb_x);
        copy_tile(cb_x, i, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_x, D + 1);
}
