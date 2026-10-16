// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISC: per compact tile, the sum of its chips copies in chip order 0, 1, ... (fp32 dest, exact bf16 copies, fp32
// SFPU adds) rounded to bf16 through cb_mid (the rounding of the all-reduce result), then plus the residual's rows
// packed like the compact tile -> one bf16 tile: the row all-reduce and the residual add, identical on every chip.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mid = get_compile_time_arg_val(1);
    constexpr uint32_t cb_res = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t tiles = get_compile_time_arg_val(4);
    constexpr uint32_t chips = get_compile_time_arg_val(5);

    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_res, tiles);
    for (uint32_t t = 0; t < tiles; t++) {
        cb_wait_front(cb_in, chips);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_in);
        copy_tile(cb_in, 0, 0);
        for (uint32_t c = 1; c < chips; c++) {
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, c, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_in, chips);

        cb_reserve_back(cb_mid, 1);
        tile_regs_wait();
        pack_tile(0, cb_mid);
        tile_regs_release();
        cb_push_back(cb_mid, 1);

        cb_wait_front(cb_mid, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_res);
        copy_tile(cb_res, t, 0);
        copy_tile_to_dst_init_short(cb_mid);
        copy_tile(cb_mid, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_mid, 1);

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_res, tiles);
}
