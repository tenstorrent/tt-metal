// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode gate|up, bank-local form (TRISC). Roles as in gu_local_reader.cpp:
//   1 bank reader: per routed expert, nr partial outputs x_b . W_b[:, n] over K slice b (custom_mm, 1x32 activation
//     tiles, LoFi, fp32 dest), packed as fp32 1x32 tiles to cb_part.
//   2 reducer, per unit (slot s, pair j) on the core: g = sum over banks of the partial gate tiles, u likewise
//     (SFPU adds over the result rows), h = act(g, u) * w_s (GeGLU, or the gpt-oss SwiGLU under SWIGLU; w_s from
//     BRISC through the mailbox), packed to cb_hout.
// compile-time args: [cb_x, cb_w, cb_part, cb_recv, cb_hout, (unused), nr, k, banks, cb_bias]
// HAS_BIAS: the reducer adds the expert's gate / up bias rows before the activation.
// runtime args: [role, ks (bank reader: this bank's K tiles), units (reducer)]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#ifdef SWIGLU
#include "api/compute/eltwise_unary/clamp.h"
#endif

constexpr uint32_t kRowIters = 2;  // a 1x32 row occupies the first rows of DST faces 0 and 1

ALWI void gelu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gelu, (false, DST_ACCUM_MODE, kRowIters), idst, VectorMode::R));
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

ALWI void add_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, kRowIters, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}

ALWI void scale_rows(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, kRowIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}

// gate in DST g, up in DST u -> activation in DST g
#ifdef SWIGLU
// gpt-oss: (clamp(u, -L, L) + 1) * g' * sigmoid(alpha g'), g' = min(g, L); compile-time float bits
constexpr uint32_t kLimit = SWIGLU_LIMIT, kNegLimit = SWIGLU_NEG_LIMIT, kNegBig = 0xff7fffffu;
constexpr uint32_t kAlpha = SWIGLU_ALPHA, kInvAlpha = SWIGLU_INV_ALPHA, kOne = 0x3f800000u;
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
ALWI void activation(uint32_t g, uint32_t u) {
    clamp_tile_init();
    clamp_rows(g, kNegBig, kLimit);
    clamp_rows(u, kNegLimit, kLimit);
    binop_with_scalar_tile_init();
    scalar_rows<MUL_UNARY>(g, kAlpha);
    scalar_rows<ADD_UNARY>(u, kOne);
    silu_tile_init();
    silu_rows(g);
    binop_with_scalar_tile_init();
    scalar_rows<MUL_UNARY>(g, kInvAlpha);
    mul_binary_tile_init();
    mul_rows(g, u, g);
}
#else
// Gemma: gelu(g) * u
ALWI void activation(uint32_t g, uint32_t u) {
    gelu_tile_init<false>();
    gelu_rows(g);
    mul_binary_tile_init();
    mul_rows(g, u, g);
}
#endif

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_hout = get_compile_time_arg_val(4);
    constexpr uint32_t nr = get_compile_time_arg_val(6);
    constexpr uint32_t k = get_compile_time_arg_val(7);
    constexpr uint32_t banks = get_compile_time_arg_val(8);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(9);
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t ks = get_arg_val<uint32_t>(1);
    const uint32_t units = get_arg_val<uint32_t>(2);
    constexpr uint32_t dst_group = 4;  // output tiles held in DST at once (fp32 accumulation)
    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    if (role == 1) {
        custom_mm_block_init<transpose, dest_layout>(cb_x, cb_w, cb_part);
        cb_wait_front(cb_x, ks);
        for (uint32_t s = 0; s < k; ++s) {
            cb_wait_front(cb_w, nr * ks);
            for (uint32_t n0 = 0; n0 < nr; n0 += dst_group) {
                const uint32_t n1 = n0 + dst_group < nr ? n0 + dst_group : nr;
                tile_regs_acquire();
                for (uint32_t n = n0; n < n1; ++n) {
                    custom_mm_block<dest_layout>(cb_x, cb_w, 0, n * ks, n - n0, ks);
                }
                tile_regs_commit();
                cb_reserve_back(cb_part, n1 - n0);
                tile_regs_wait();
                for (uint32_t n = n0; n < n1; ++n) {
                    pack_tile(n - n0, cb_part);
                }
                tile_regs_release();
                cb_push_back(cb_part, n1 - n0);
            }
            cb_pop_front(cb_w, nr * ks);
        }
        cb_pop_front(cb_x, ks);
        custom_mm_block_uninit<dest_layout>(cb_part);
    } else if (role == 2) {
        compute_kernel_hw_startup(cb_recv, cb_hout);
        cb_wait_front(cb_recv, 2 * banks * units);
        for (uint32_t u = 0; u < units; ++u) {
            uint32_t score = 0;
            MATH(score = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
            const uint32_t base = u * 2 * banks;
            tile_regs_acquire();
            copy_init(cb_recv);
            copy_tile(cb_recv, base, 0);
            copy_tile(cb_recv, base + 1, 1);
            for (uint32_t b = 1; b < banks; ++b) {
                copy_init(cb_recv);
                copy_tile(cb_recv, base + 2 * b, 2);
                copy_tile(cb_recv, base + 2 * b + 1, 3);
                add_binary_tile_init();
                add_rows(0, 2, 0);
                add_rows(1, 3, 1);
            }
#ifdef HAS_BIAS
            // the expert's gate and up bias rows (bf16, from the writer)
            cb_wait_front(cb_bias, 2);
            reconfig_data_format_srca(cb_bias);
            copy_init(cb_bias);
            copy_tile(cb_bias, 0, 2);
            copy_tile(cb_bias, 1, 3);
            add_binary_tile_init();
            add_rows(0, 2, 0);
            add_rows(1, 3, 1);
            cb_pop_front(cb_bias, 2);
            reconfig_data_format_srca(cb_recv);
#endif
            activation(0, 1);
            binop_with_scalar_tile_init();
            scale_rows(0, score);
            tile_regs_commit();
            cb_reserve_back(cb_hout, 1);
            tile_regs_wait();
            pack_tile(0, cb_hout);
            tile_regs_release();
            cb_push_back(cb_hout, 1);
        }
        cb_pop_front(cb_recv, 2 * banks * units);
    }
}
