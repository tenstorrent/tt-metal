// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(0), stride = get_arg_val<uint32_t>(1);
    const uint32_t kt = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(2, 3, 4);
    cb_wait_front(0, 1);
    const uint32_t live = read_tile_value(0, 0, 0);
    for (uint32_t key_tile = first; key_tile < kt; key_tile += stride) {
        if (key_tile * 32 >= live) {
            continue;
        }
        cb_wait_front(3, 4);
        for (uint32_t head = 0; head < 4; ++head) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < 4; ++w) {
                cb_wait_front(2, 1);
                unary_bcast_init<BroadcastType::ROW>(2);
                unary_bcast<BroadcastType::ROW>(2, 0, 0);
                unary_bcast_uninit<BroadcastType::ROW>(2);
                // Restore ordinary key-tile copying after the row broadcast.
                copy_init(3);
                copy_tile(3, w, 1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, w == 0 ? 2 : 0);
                if (w != 0) {
                    add_binary_tile_init();
                    add_binary_tile(2, 0, 2);
                }
                cb_pop_front(2, 1);
            }
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
            cb_reserve_back(4, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(2, 4);
            tile_regs_release();
            cb_push_back(4, 1);
        }
        // All four heads have consumed this same key block.
        cb_pop_front(3, 4);
    }
    cb_pop_front(0, 1);
}
