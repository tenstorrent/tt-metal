// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute for conv_hm: conv = sum_j tap[j] * state_new[j] over the shifted window (state_new[j] = old state
// row j+1 for j < K-1, the new row for j = K-1), then SiLU, all in fp32 DST (SFPU dest-to-dest multiply/add, so
// the sum accumulates without the bf16 rounding the ttnn mul/mac chain does after every tap).

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_new = 0, cb_cst = 1, cb_tap = 2, cb_out = 3;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(cb_cst, cb_tap, cb_out);
    CircularBuffer cn(cb_new), cc(cb_cst), ct(cb_tap), co(cb_out);
    cn.wait_front(1);
    cc.wait_front(K - 1);
    ct.wait_front(K);
    co.reserve_back(1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < K; j++) {
        const uint32_t d = (j == 0) ? 0u : 1u;  // tap 0 seeds DST[0]; later terms go to DST[1] then add
        if (j + 1 < K) {
            copy_init(cb_cst);
            copy_tile(cb_cst, j, d);
        } else {
            copy_init(cb_new);
            copy_tile(cb_new, 0, d);
        }
        copy_init(cb_tap);
        copy_tile(cb_tap, j, 2);
        mul_binary_tile_init();
        mul_binary_tile(d, 2, d);
        if (j > 0) {
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
    }
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    co.push_back(1);
    cn.pop_front(1);
    cc.pop_front(K - 1);
    ct.pop_front(K);
}
