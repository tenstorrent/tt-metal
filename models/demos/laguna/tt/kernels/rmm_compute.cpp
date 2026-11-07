// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Router logits, K-split, compute: DST m = sum_k x[m, k] @ w[k, n] over this core's Kc tiles (fp32 DST), packed
// fp32 to cb_part. A column's reducer (reduce = 1) then adds the other KC - 1 chunks' partials (unpacked straight to
// DST in fp32: exact sums), applies the sigmoid and packs the scores to cb_out.

#include <cstdint>
#include "api/compute/reg_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kc = get_compile_time_arg_val(1);
    constexpr uint32_t KC = get_compile_time_arg_val(2);
    constexpr uint32_t reduce = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_part = 2, cb_recv = 3, cb_out = 16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_part);
    matmul_init(cb_x, cb_w);
    cb_wait_front(cb_x, Mt * Kc);
    cb_wait_front(cb_w, Kc);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kc; ++k) {
        for (uint32_t m = 0; m < Mt; ++m) {
            matmul_tiles(cb_x, cb_w, m * Kc + k, k, m);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_part);
    cb_reserve_back(cb_part, Mt);
    for (uint32_t m = 0; m < Mt; ++m) {
        pack_tile(m, cb_part);
    }
    cb_push_back(cb_part, Mt);
    tile_regs_release();
    cb_pop_front(cb_x, Mt * Kc);
    cb_pop_front(cb_w, Kc);
    if constexpr (reduce) {
        cb_wait_front(cb_part, Mt);
        cb_wait_front(cb_recv, (KC - 1) * Mt);
        copy_init(cb_part);
        reconfig_data_format_srca(cb_part);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, Mt);
        for (uint32_t m = 0; m < Mt; ++m) {
            tile_regs_acquire();
            copy_init(cb_part);
            copy_tile(cb_part, m, 0);
            copy_init(cb_recv);
            for (uint32_t i = 0; i < KC - 1; ++i) {
                copy_tile(cb_recv, i * Mt + m, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            sigmoid_tile_init();
            sigmoid_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, Mt);
        cb_pop_front(cb_part, Mt);
        cb_pop_front(cb_recv, (KC - 1) * Mt);
    }
}
