// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert gate|up matmul + GPT-OSS SwiGLU (TRISC) for the decode MoE stream (experts/stream.py).
//
// For each routed expert and each (gate, up) column pair this core owns:
//   g = x . W[:, gate col] + b_g,  u = x . W[:, up col] + b_u   (custom_mm, 1x32 activation tile, LoFi;
//                                                                 the bias rides in the last K tile)
//   act = (clamp(u, -limit, limit) + 1) * g' * sigmoid(alpha * g'),  g' = min(g, limit)
// computed in DST with SFPU ops (g' * sigmoid(alpha g') = silu(alpha g') / alpha), scaled by the expert's routing
// score w_e (received from BRISC through the mailbox) and packed as one 1x32 tile.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

// SFPU ops over the result rows only: a 1x32 matmul result occupies the first rows of DST faces 0 and 1
// (VectorMode::R, 2 iterations, as the DeepSeek DRAM-streaming matmul does for tile heights <= 4).
constexpr uint32_t kRowIters = 2;
ALWI void clamp_rows(uint32_t idst, uint32_t lo, uint32_t hi) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_clamp, (APPROX, kRowIters), idst, VectorMode::R, lo, hi));
}
template <int OP>
ALWI void scalar_rows(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, kRowIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}
ALWI void silu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, kRowIters), idst, VectorMode::R));
}
ALWI void mul_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, kRowIters, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t pairs = get_compile_time_arg_val(4);
    constexpr uint32_t num_sel = get_compile_time_arg_val(5);
    constexpr uint32_t limit = get_compile_time_arg_val(6);       // float bits
    constexpr uint32_t neg_limit = get_compile_time_arg_val(7);   // float bits
    constexpr uint32_t neg_big = get_compile_time_arg_val(8);     // float bits (no lower clamp on the gate)
    constexpr uint32_t alpha = get_compile_time_arg_val(9);       // float bits
    constexpr uint32_t inv_alpha = get_compile_time_arg_val(10);  // float bits
    constexpr uint32_t one = get_compile_time_arg_val(11);        // float bits

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);

    uint32_t scores[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        scores[e] = 0;
        MATH(scores[e] = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    }

    cb_wait_front(cb_x, kt);
    for (uint32_t e = 0; e < num_sel; ++e) {
        for (uint32_t p = 0; p < pairs; ++p) {
            tile_regs_acquire();
            cb_wait_front(cb_w, kt);
            custom_mm_block<true>(cb_x, cb_w, 0, 0, 0, kt);
            cb_pop_front(cb_w, kt);
            cb_wait_front(cb_w, kt);
            custom_mm_block<true>(cb_x, cb_w, 0, 0, 1, kt);
            cb_pop_front(cb_w, kt);

            clamp_tile_init();
            clamp_rows(0, neg_big, limit);
            clamp_rows(1, neg_limit, limit);
            binop_with_scalar_tile_init();
            scalar_rows<MUL_UNARY>(0, alpha);
            scalar_rows<ADD_UNARY>(1, one);
            silu_tile_init();
            silu_rows(0);
            binop_with_scalar_tile_init();
            scalar_rows<MUL_UNARY>(0, inv_alpha);
            mul_binary_tile_init();
            mul_rows(0, 1, 0);
            binop_with_scalar_tile_init();
            scalar_rows<MUL_UNARY>(0, scores[e]);
            tile_regs_commit();

            cb_reserve_back(cb_act, 1);
            tile_regs_wait();
            pack_tile(0, cb_act);
            tile_regs_release();
            cb_push_back(cb_act, 1);

            custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);
        }
    }
    cb_pop_front(cb_x, kt);
    custom_mm_block_uninit<dense_packing>();
}
