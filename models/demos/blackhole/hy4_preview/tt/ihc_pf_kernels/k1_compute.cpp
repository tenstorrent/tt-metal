// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K1 compute: M = sum_{s,c} X_{s,c} @ W_{s,c} (dest 0, fp32 accumulate, HiFi4), Q = sum_{s,c} X_{s,c}^2
// (dest 1, SFPU adds of the FPU squares); one dest section for the whole core. Output 2 tiles: [M, Q].  CT: 0 cb_x, 1
// cb_w, 2 cb_out, 3 NB (tiles per operand), 4 B.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t NB = get_compile_time_arg_val(3);
    constexpr uint32_t B = get_compile_time_arg_val(4);
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    CircularBuffer x(cb_x), w(cb_w), out(cb_out);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    fill_tile(1, 0.0f);
    for (uint32_t b = 0; b < NB / B; ++b) {
        x.wait_front(B);
        w.wait_front(B);
        reconfig_data_format(cb_w, cb_x);
        matmul_init(cb_x, cb_w);
        for (uint32_t i = 0; i < B; ++i) {
            matmul_tiles(cb_x, cb_w, i, i, 0);
        }
        reconfig_data_format_srca(cb_x);
        copy_tile_to_dst_init_short(cb_x);
        for (uint32_t i = 0; i < B; ++i) {
            copy_tile(cb_x, i, 2 + i);
        }
        square_tile_init();
        for (uint32_t i = 0; i < B; ++i) {
            square_tile(2 + i);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < B; ++i) {
            add_binary_tile(1, 2 + i, 1);
        }
        x.pop_front(B);
        w.pop_front(B);
    }
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(2);
    pack_tile(0, cb_out);
    pack_tile(1, cb_out);
    out.push_back(2);
    tile_regs_release();
}
