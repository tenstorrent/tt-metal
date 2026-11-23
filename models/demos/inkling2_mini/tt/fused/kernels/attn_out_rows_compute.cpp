// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// attn_out_compute.cpp for the T tokens of a verify pass (attn_out_rows_reader.cpp): the same ops per token, each
// token's pages popped after use. Fused attention output prep, compute (fp32 SFPU math, one dense page):
//   out = bf16(attn * sigmoid(gate))
// merge: attn = exact near / far merge  (A r + o_near (1 - a)) / (a r + 1 - a)   (attention_fast.py module doc)
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, 0, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t merge = get_named_compile_time_arg_val("merge");
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_m = 2, cb_r = 3, cb_o = 4, cb_out = 16;
    constexpr uint32_t one = 0x3f800000u, minus_one = 0xbf800000u;
    // fr: the writer reads elements 0 .. 383 (3 heads) of the page, faces 0 and 1: run the SFPU ops on those only
    constexpr VectorMode VM = get_named_compile_time_arg_val("fr") ? VectorMode::R : VectorMode::RC;
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    compute_kernel_hw_startup(cb_a, cb_out);
    for (uint32_t t = 0; t < tokens; ++t) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_g, 1);
        tile_regs_acquire();
        load(cb_a, 0);
        load(cb_g, 4);
        if constexpr (merge) {
            cb_wait_front(cb_m, 1);
            cb_wait_front(cb_r, 1);
            cb_wait_front(cb_o, 1);
            load(cb_m, 1);
            load(cb_r, 2);
            load(cb_o, 3);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0, VM);  // A r
            mul_binary_tile(1, 2, 5, VM);  // a r
            binop_with_scalar_tile_init();
            mul_unary_tile(1, minus_one, VM);
            add_unary_tile(1, one, VM);  // w = 1 - a
            mul_binary_tile_init();
            mul_binary_tile(3, 1, 3, VM);  // o_near w
            add_binary_tile_init();
            add_binary_tile(0, 3, 0, VM);  // num
            add_binary_tile(5, 1, 5, VM);  // den
            recip_tile_init();
            recip_tile(5, VM);
            mul_binary_tile_init();
            mul_binary_tile(0, 5, 0, VM);
        }
        sigmoid_tile_init();
        sigmoid_tile<VM>(4);
        mul_binary_tile_init();
        mul_binary_tile(0, 4, 0, VM);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_g, 1);
        if constexpr (merge) {
            cb_pop_front(cb_m, 1);
            cb_pop_front(cb_r, 1);
            cb_pop_front(cb_o, 1);
        }
    }
}
