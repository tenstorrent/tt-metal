// SPDX-License-Identifier: Apache-2.0
// tp32pf floor microbench TRISC: per owned expert, per pass: dest[MT x 4] += x[b] (MT x 24) @ w[b] (24 x 4) over 8
// banks, pack MT*4 tiles. CT: 0 cb_x, 1 cb_w, 2 cb_o, 3 MT, 4 PASSES, 5 NE, 6 NC. RT: 0 core idx
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t MT = get_compile_time_arg_val(3);
    constexpr uint32_t PASSES = get_compile_time_arg_val(4);
    constexpr uint32_t NE = get_compile_time_arg_val(5);
    constexpr uint32_t NC = get_compile_time_arg_val(6);
    const uint32_t me = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o);
    matmul_block_init(cb_x, cb_w, false, 4, MT, 24);
    cb_wait_front(cb_x, MT * 192);
    for (uint32_t e = me; e < NE; e += NC) {
        for (uint32_t p = 0; p < PASSES; ++p) {
            cb_reserve_back(cb_o, MT * 4);
            tile_regs_acquire();
            for (uint32_t b = 0; b < 8; ++b) {
                cb_wait_front(cb_w, 96);
                for (uint32_t k = 0; k < 24; ++k) {
                    matmul_block(cb_x, cb_w, ((b + me) & 7) * MT * 24 + k, k * 4, 0, false, 4, MT, 24);
                }
                cb_pop_front(cb_w, 96);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < MT * 4; ++i) {
                pack_tile(i, cb_o, i);
            }
            tile_regs_release();
            cb_push_back(cb_o, MT * 4);
        }
    }
}
