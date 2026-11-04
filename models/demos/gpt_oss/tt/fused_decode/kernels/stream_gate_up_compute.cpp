// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert gate|up matmul + GPT-OSS SwiGLU (TRISC) for the decode MoE stream (fused_decode/stream.py).
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
#if defined(EXP_TAIL_SUBS) && defined(EXPERT_SPLIT)
#error "EXP_TAIL_SUBS: the plain BFP4 / BFP8 streams only"
#endif
#ifdef EXPERT_SPLIT
// TEMP experiment (GPT_OSS_DECODE_EXPERTS_SPLIT=1, fused_decode/mxfp4_split.py): exact MXFP4 split weights, two
// compressed custom_mm calls per column (BFP4 small + bias from cb_w, BFP2 big from cb_w2) into the same DST tile.
#include "api/compute/experimental/compressed_custom_mm.h"
#include "split_meta.hpp"
#endif

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
#ifdef W_FP32_FOLD
// split_acc keeps two FP32 accumulators per output row (dest rows 0 / 8 of each face: alternate K faces);
// custom_mm's own finalize moves one through SrcB (TF32, ~10-bit mantissa) before adding it. custom_mm_block<false>
// leaves them apart and this folds them in FP32 on the SFPU: rows 8.. of each face onto rows 0.. (VectorMode::R).
#ifdef TRISC_MATH
#include "sfpi.h"
namespace ckernel::sfpu {
template <int ITERATIONS>
inline void calculate_split_fold() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat lo = sfpi::dst_reg[0];
        const sfpi::vFloat hi = sfpi::dst_reg[4];  // 8 dest rows further
        sfpi::dst_reg[0] = lo + hi;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
ALWI void fold_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_split_fold, (2), idst, VectorMode::R));
}
#endif

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

#ifdef EXPERT_SPLIT
    constexpr uint32_t cb_w2 = get_compile_time_arg_val(12);
    constexpr uint32_t kt4 = get_compile_time_arg_val(13);
    constexpr uint32_t kt2 = get_compile_time_arg_val(14);
    constexpr uint32_t n4 = get_compile_time_arg_val(15);
    constexpr uint32_t n2 = get_compile_time_arg_val(16);
    static constexpr auto meta4 = split_meta::cta_array<17, n4>();
    static constexpr auto meta2 = split_meta::cta_array<17 + n4, n2>();
    const uint32_t meta4_addr = reinterpret_cast<uint32_t>(meta4.data());
    const uint32_t meta2_addr = reinterpret_cast<uint32_t>(meta2.data());
    constexpr bool transpose = true;
    constexpr bool split_acc = false;
    constexpr bool dense_packing = false;
#if defined(SPLIT_RANKS) && defined(SPLIT_REST_PKG)
    // inexact ranks read the package's bfloat4_b tiles in their normal orientation (no transpose on unpack)
    constexpr uint32_t split_ranks = SPLIT_RANKS;
    if (split_ranks > 0) {
        compressed_custom_mm_block_init<true, split_acc, dense_packing>(cb_x, cb_w, cb_act);
    } else {
        compressed_custom_mm_block_init<false, split_acc, dense_packing>(cb_x, cb_w, cb_act);
    }
#else
    compressed_custom_mm_block_init<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);
#endif
#else
#ifdef W_TRANSPOSED
    // transposed BFP4 tiles, 16-value groups along K (GPT_OSS_DECODE_SPLIT_NATIVE: a rank-0 split stream's rounded
    // parts)
    constexpr bool transpose = true;
#else
    constexpr bool transpose = false;
#endif
#ifdef W_NO_SPLIT_ACC
    constexpr bool split_acc = false;
#else
    constexpr bool split_acc = true;
#endif
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);
#endif

    uint32_t scores[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        scores[e] = 0;
        MATH(scores[e] = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    }

    cb_wait_front(cb_x, kt);
    for (uint32_t e = 0; e < num_sel; ++e) {
        for (uint32_t p = 0; p < pairs; ++p) {
            tile_regs_acquire();
#ifdef EXPERT_SPLIT
            for (uint32_t d = 0; d < 2; ++d) {  // gate column -> DST 0, up column -> DST 1
#ifdef SPLIT_RANKS
                // routed ranks >= SPLIT_RANKS: the rounded BFP4 tiles alone (same tile count and formats as meta4)
                const bool exact = e < SPLIT_RANKS;
#else
                constexpr bool exact = true;
#endif
                cb_wait_front(cb_w, 1);
                compressed_custom_mm_block<false>(cb_x, cb_w, meta4_addr, d, kt4);
                cb_pop_front(cb_w, 1);
                if (exact) {
                    cb_wait_front(cb_w2, 1);
                    compressed_custom_mm_block<false>(cb_x, cb_w2, meta2_addr, d, kt2);
                    cb_pop_front(cb_w2, 1);
                }
            }
#else
#ifdef W_FP32_FOLD
            constexpr bool finalize = false;  // fold_rows below
#else
            constexpr bool finalize = true;
#endif
            cb_wait_front(cb_w, kt);
            custom_mm_block<finalize>(cb_x, cb_w, 0, 0, 0, kt);
            cb_pop_front(cb_w, kt);
#ifdef EXP_TAIL_SUBS
            // TEMP experiment (fused_decode/config.py DECODE_EXPERT_TAIL_SPLIT): the last column, pushed by the reader
            // in EXP_TAIL_SUBS sub-blocks, is multiplied sub-block by sub-block as they land. custom_mm accumulates
            // every K tile into the same DST rows whatever the call split (each tile's two K faces always go to the
            // same split accumulators), and only the last call finalizes: the same MVMUL / ELWADD sequence as one call.
            if (e + 1 == num_sel && p + 1 == pairs) {
                constexpr uint32_t tt = EXP_TAIL_TILES;
                for (uint32_t t = 0; t + 1 < EXP_TAIL_SUBS; ++t) {
                    cb_wait_front(cb_w, (t + 1) * tt);
                    custom_mm_block<false>(cb_x, cb_w, t * tt, t * tt, 1, tt);
                }
                cb_wait_front(cb_w, kt);
                custom_mm_block<finalize>(cb_x, cb_w, kt - tt, kt - tt, 1, tt);
                cb_pop_front(cb_w, kt);
            } else
#endif
            {
                cb_wait_front(cb_w, kt);
                custom_mm_block<finalize>(cb_x, cb_w, 0, 0, 1, kt);
                cb_pop_front(cb_w, kt);
            }
#ifdef W_FP32_FOLD
            binop_with_scalar_tile_init();
            fold_rows(0);
            fold_rows(1);
#endif
#endif

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

#if defined(EXPERT_SPLIT) && defined(SPLIT_RANKS) && defined(SPLIT_REST_PKG)
            {
                const uint32_t next_e = p + 1 == pairs ? e + 1 : e;  // the expert of the next column pair
                if (next_e < split_ranks) {
                    compressed_custom_mm_block_init_short<true, split_acc, dense_packing>(cb_x, cb_w, cb_act);
                } else {
                    compressed_custom_mm_block_init_short<false, split_acc, dense_packing>(cb_x, cb_w, cb_act);
                }
            }
#elif defined(EXPERT_SPLIT)
            compressed_custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);
#else
            custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);
#endif
        }
    }
    cb_pop_front(cb_x, kt);
#ifdef EXPERT_SPLIT
    compressed_custom_mm_block_uninit<dense_packing>();
#else
    custom_mm_block_uninit<dense_packing>();
#endif
}
