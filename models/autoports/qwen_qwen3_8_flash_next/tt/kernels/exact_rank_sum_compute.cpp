// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"

// The accurate H reducer first sums rows j,j+4,j+8,j+12, then adds
// rows j+16,j+20,j+24,j+28. With four live rank rows, this is (rj+0)+0.
// Keep BOTH FP32 SFPU additions: omitting them changes negative-zero bits.
ALWI void widen_rank_with_padding(uint32_t rank, uint32_t dst) {
    copy_init(0);
    copy_tile(0, rank, dst);
    add_binary_tile_init();
    add_binary_tile(dst, 2, dst);
    add_binary_tile(dst, 2, dst);
}
void kernel_main() {
    compute_kernel_hw_startup(0, 0, 1);
    const uint32_t count = get_arg_val<uint32_t>(0);
    for (uint32_t tile = 0; tile < count; ++tile) {
        cb_wait_front(0, 4);
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(2, 0.0f);
        widen_rank_with_padding(2, 0);
        widen_rank_with_padding(3, 1);
        add_binary_tile(0, 1, 0);
        widen_rank_with_padding(1, 1);
        add_binary_tile(1, 0, 0);
        widen_rank_with_padding(0, 1);
        add_binary_tile(1, 0, 0);
        cb_reserve_back(1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 1);
        tile_regs_release();
        cb_push_back(1, 1);
        cb_pop_front(0, 4);
    }
}
