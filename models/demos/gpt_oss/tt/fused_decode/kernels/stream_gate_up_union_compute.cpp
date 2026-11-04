// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Union-of-experts gate|up matmul + GPT-OSS SwiGLU (TRISC) for the multi-row decode MoE (DFlash verify;
// fused_decode/union.py ExpertGateUpUnion). As kernels/stream_gate_up_compute.cpp on [rows, 32] activation tiles, for
// the U union experts (count from BRISC through the mailboxes); each SwiGLU row is scaled by that row's routing
// weight of the expert: tile i of cb_sc holds w[r][i] in row r, loaded into DST 2 by a 1-tile matmul with the
// identity tile (exact: BFP4 0 / 1, the weight format, so no unpacker format switch) and multiplied in.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

constexpr uint32_t kRowIters = ROWS <= 4 ? 2 : ROWS / 2;
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
    constexpr uint32_t limit = get_compile_time_arg_val(5);      // float bits
    constexpr uint32_t neg_limit = get_compile_time_arg_val(6);  // float bits
    constexpr uint32_t neg_big = get_compile_time_arg_val(7);    // float bits (no lower clamp on the gate)
    constexpr uint32_t alpha = get_compile_time_arg_val(8);      // float bits
    constexpr uint32_t inv_alpha = get_compile_time_arg_val(9);  // float bits
    constexpr uint32_t one = get_compile_time_arg_val(10);       // float bits
    constexpr uint32_t cb_sc = get_compile_time_arg_val(11);
    constexpr uint32_t cb_id = get_compile_time_arg_val(12);

    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, dest_layout>(cb_x, cb_w, cb_act);

    uint32_t num_sel = 0;
    UNPACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(num_sel = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    cb_wait_front(cb_x, kt);
    cb_wait_front(cb_id, 1);
    // one union expert (processing index e) from weight CB wcb
    auto expert = [&](uint32_t e, uint32_t wcb) {
        for (uint32_t p = 0; p < pairs; ++p) {
            tile_regs_acquire();
            cb_wait_front(wcb, kt);
            custom_mm_block<dest_layout>(cb_x, wcb, 0, 0, 0, kt);
            cb_pop_front(wcb, kt);
            cb_wait_front(wcb, kt);
            custom_mm_block<dest_layout>(cb_x, wcb, 0, 0, 1, kt);
            cb_pop_front(wcb, kt);
            custom_mm_block_init_short<transpose, dest_layout>(cb_sc, cb_id, cb_act);
            custom_mm_block<dest_layout>(cb_sc, cb_id, e, 0, 2, 1);  // row weights

#ifdef DENSE_SILU
            // Plain SwiGLU (dense MLP, e.g. the DFlash drafter): silu(g) * u.
            silu_tile_init();
            silu_rows(0);
#else
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
#endif
            mul_binary_tile_init();
            mul_rows(0, 1, 0);
            mul_rows(0, 2, 0);
            tile_regs_commit();

            cb_reserve_back(cb_act, 1);
            tile_regs_wait();
            pack_tile(0, cb_act);
            tile_regs_release();
            cb_push_back(cb_act, 1);

            custom_mm_block_init_short<transpose, dest_layout>(cb_x, wcb, cb_act);
        }
    };
    uint32_t e = 0;
#ifdef MOE_PRED
    // the P predicted experts' columns come in MOE_PRED_CB (prefetched by the reader before the union was known);
    // hit bit k (writer, mailbox): prediction k was routed and is processed now, else its columns are skipped
    uint32_t hits = 0;
    UNPACK(hits = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(hits = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(hits = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    for (uint32_t k = 0; k < MOE_PRED; ++k) {
        if ((hits >> k) & 1) {
            cb_wait_front(cb_sc, e + 1);
            custom_mm_block_init_short<transpose, dest_layout>(cb_x, MOE_PRED_CB, cb_act);
            expert(e, MOE_PRED_CB);
            ++e;
        } else {
            cb_wait_front(MOE_PRED_CB, 2 * pairs * kt);
            cb_pop_front(MOE_PRED_CB, 2 * pairs * kt);
        }
    }
    custom_mm_block_init_short<transpose, dest_layout>(cb_x, cb_w, cb_act);
#endif
    for (; e < num_sel; ++e) {
        cb_wait_front(cb_sc, e + 1);  // the writer pushes the scale tiles one by one
        expert(e, cb_w);
    }
    cb_pop_front(cb_x, kt);
    cb_pop_front(cb_sc, num_sel);
    cb_pop_front(cb_id, 1);
    custom_mm_block_uninit<dest_layout>(cb_act);
}
