// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B17 fused attention back: compute. out = (sum_k x_k w_k) * sigmoid(g) (GFOLD=1) or * g (GFOLD=0), one output tile
// per core, HiFi4 fp32 dest accumulation. CT: 0 cb_x, 1 cb_w, 2 cb_g, 3 cb_out, 4 KT, 5 GFOLD
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t KT = get_compile_time_arg_val(4);
    constexpr bool GFOLD = get_compile_time_arg_val(5) != 0;
    constexpr bool SIGFAST = get_compile_time_arg_val(7) != 0;
    constexpr bool ROUND = get_compile_time_arg_val(6) != 0;
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    CircularBuffer x(cb_x), w(cb_w), g(cb_g), o(cb_out);
    x.wait_front(KT);
    w.wait_front(KT);
    g.wait_front(1);
    reconfig_data_format(cb_x, cb_w);
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    for (uint32_t k = 0; k < KT; ++k) {
        matmul_tiles(cb_x, cb_w, k, k, 0);
    }
    if constexpr (ROUND) {  // the stock matmul output is a bf16 tensor: round the accumulator to bf16 before the gate
                            // multiply
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
    }
    reconfig_data_format_srca(cb_w, cb_g);
    copy_init(cb_g);
    copy_tile(cb_g, 0, 1);
    if constexpr (GFOLD) {
        sigmoid_tile_init<SIGFAST>();
        sigmoid_tile<VectorMode::RC, SIGFAST>(1);
        if constexpr (ROUND) {  // the stock input activation is applied to a bf16 operand: its result is a bf16 value
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(1);
        }
    }
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    o.reserve_back(1);
    pack_tile(0, cb_out);
    o.push_back(1);
    tile_regs_release();
    x.pop_front(KT);
    w.pop_front(KT);
    g.pop_front(1);
}
