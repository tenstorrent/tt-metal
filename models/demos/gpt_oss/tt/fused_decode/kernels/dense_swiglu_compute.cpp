// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batched decode dense MoE, compute (dense_swiglu_reader.cpp): per column tile j of this core's shard
//   g = gate_j + gate_bias_j, u = up_j + up_bias_j (row broadcast of the bias rows),
//   act_j = SwiGLU(g, u) (gpt-oss: (clamp(u, -l, l) + 1) clamp(g, max l) sigmoid(alpha clamp(g, max l)),
//   gmoe_swiglu.hpp, in the same SFPU pass as the multiply)
//           x (w E_j) (the routing weight of each column's expert: the [32 tokens, E] weights times the one-hot expand
//           tile, a K = 1 matmul)
// packed into the act shard (a tensor-backed CB) in order.
//
// compile-time args: [cb_gate, cb_up, cb_e, cb_w, cb_bg, cb_ub, cb_act, NT]

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "gmoe_swiglu.hpp"

void kernel_main() {
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_e = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_bg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ub = get_compile_time_arg_val(5);
    constexpr uint32_t cb_act = get_compile_time_arg_val(6);
    constexpr uint32_t NT = get_compile_time_arg_val(7);

    compute_kernel_hw_startup(cb_gate, cb_bg, cb_act);
    cb_wait_front(cb_w, 1);
    cb_wait_front(cb_gate, NT);
    cb_wait_front(cb_up, NT);
    for (uint32_t j = 0; j < NT; ++j) {
        cb_wait_front(cb_bg, 1);
        cb_wait_front(cb_ub, 1);
        cb_wait_front(cb_e, 1);
        tile_regs_acquire();
        // (the inits do not switch the unpacker formats: state_configure is a no-op outside the sentinel build)
        reconfig_data_format(cb_gate, cb_bg);
        add_bcast_rows_init(cb_gate, cb_bg);
        add_tiles_bcast_rows(cb_gate, cb_bg, j, 0, 0);
        add_tiles_bcast_rows(cb_up, cb_ub, j, 0, 1);
        reconfig_data_format(cb_e, cb_w);  // matmul: SrcA <- in1 (expand), SrcB <- in0 (weights)
        matmul_init(cb_w, cb_e, 0);
        matmul_tiles(cb_w, cb_e, 0, 0, 2);
        MATH((ckernel::gmoe_swiglu_init()));
        MATH((ckernel::gmoe_swiglu_weighted<DST_ACCUM_MODE>(0, 1, 2, 0)));  // (times the weights tile: one SFPU pass)
        tile_regs_commit();
        cb_reserve_back(cb_act, 1);
        tile_regs_wait();
        pack_tile(0, cb_act);
        tile_regs_release();
        cb_push_back(cb_act, 1);
        cb_pop_front(cb_bg, 1);
        cb_pop_front(cb_ub, 1);
        cb_pop_front(cb_e, 1);
    }
    cb_pop_front(cb_w, 1);
    cb_pop_front(cb_gate, NT);
    cb_pop_front(cb_up, NT);
}
