// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert matmuls (TRISC) of the ND-sharded-weight decode MoE (fused_decode/nd.py, kernels/nd_reader.cpp).
//
// A group is g output columns. Its weights arrive as CB slots of kb K rows x g columns, row-major [k][c] (custom_mm
// in1 block, ct_dim = g): kt / kb weight blocks then one 1-row bias block per routed expert (and column set); the
// activation tiles (1x32) are all in cb_in0 (per segment: kt weight-row tiles, then the bias multiplier tile).
// Each slot is one custom_mm call into the group's g DST tiles, accumulating over the K blocks (split accumulation
// finalized on the group's last call).
//   mode 1 (gate|up): per routed expert two column sets, gate -> DST [0, g), up -> DST [g, 2g); then GPT-OSS SwiGLU
//     act = (clamp(u, -limit, limit) + 1) * g' * sigmoid(alpha g'), g' = min(g, limit), scaled by the expert's
//     routing score (from BRISC through the mailbox) -> g output tiles per (group, expert);
//   mode 0 (down): one column set, summed over the routed experts (cb_in0 holds the k score-weighted activation
//     segments, kt + 1 tiles each) -> g output tiles per group.
//
// runtime args: [n_groups]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

// SFPU ops over the result rows only (a 1x32 matmul result occupies the first rows of DST faces 0 and 1).
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

constexpr bool kSplitAcc = get_compile_time_arg_val(14) != 0;  // split accumulation, finalized on the last K block
constexpr uint32_t kCbW2 = get_compile_time_arg_val(15);       // odd K blocks with 2 streams (nd_stream.hpp)
constexpr uint32_t kStreams = get_compile_time_arg_val(16);
constexpr auto kLayout = kSplitAcc ? CustomMmDestLayout::DefaultSplitAcc : CustomMmDestLayout::Default;

ALWI uint32_t block_cb(uint32_t cb_w, uint32_t b) { return (kStreams == 2 && (b & 1)) ? kCbW2 : cb_w; }

template <bool last>
ALWI void mm_block(uint32_t cb_in0, uint32_t cb_w, uint32_t in0_idx, uint32_t dst, uint32_t kb, uint32_t g) {
    custom_mm_block<kLayout, kSplitAcc && last>(cb_in0, cb_w, in0_idx, 0, dst, kb, g);
}

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);  // weight K rows; + 1 bias row per segment
    constexpr uint32_t kb = get_compile_time_arg_val(4);  // divides kt
    constexpr uint32_t g = get_compile_time_arg_val(5);
    constexpr uint32_t num_sel = get_compile_time_arg_val(6);
    constexpr uint32_t mode = get_compile_time_arg_val(7);
    constexpr uint32_t limit = get_compile_time_arg_val(8);       // float bits
    constexpr uint32_t neg_limit = get_compile_time_arg_val(9);   // float bits
    constexpr uint32_t neg_big = get_compile_time_arg_val(10);    // float bits (no lower clamp on the gate)
    constexpr uint32_t alpha = get_compile_time_arg_val(11);      // float bits
    constexpr uint32_t inv_alpha = get_compile_time_arg_val(12);  // float bits
    constexpr uint32_t one = get_compile_time_arg_val(13);        // float bits

    constexpr uint32_t seg = kt + 1;
    constexpr uint32_t blocks = kt / kb + 1;  // + the 1-row bias block
    static_assert((kt / kb) * kb == kt, "weight-row blocks divide K");
    constexpr uint32_t slot_tiles = kb * g;
    constexpr uint32_t in0_tiles = mode == 1 ? seg : num_sel * seg;

    const uint32_t n_groups = get_arg_val<uint32_t>(0);

    custom_mm_block_init<false, kLayout>(cb_in0, cb_w, cb_out, g);

    uint32_t scores[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        scores[e] = 0;
    }
    if constexpr (mode == 1) {
        for (uint32_t e = 0; e < num_sel; ++e) {
            MATH(scores[e] = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        }
    }

    cb_wait_front(cb_in0, in0_tiles);
    for (uint32_t gi = 0; gi < n_groups; ++gi) {
        if constexpr (mode == 1) {
            for (uint32_t e = 0; e < num_sel; ++e) {
                tile_regs_acquire();
                for (uint32_t b = 0; b < blocks; ++b) {
                    const uint32_t cbw = block_cb(cb_w, b);
                    for (uint32_t s = 0; s < 2; ++s) {
                        cb_wait_front(cbw, slot_tiles);
                        if (b + 1 == blocks) {
                            mm_block<true>(cb_in0, cbw, kt, s * g, 1, g);
                        } else {
                            mm_block<false>(cb_in0, cbw, b * kb, s * g, kb, g);
                        }
                        cb_pop_front(cbw, slot_tiles);
                    }
                }
                for (uint32_t i = 0; i < g; ++i) {
                    const uint32_t dg = i, du = g + i;
                    clamp_tile_init();
                    clamp_rows(dg, neg_big, limit);
                    clamp_rows(du, neg_limit, limit);
                    binop_with_scalar_tile_init();
                    scalar_rows<MUL_UNARY>(dg, alpha);
                    scalar_rows<ADD_UNARY>(du, one);
                    silu_tile_init();
                    silu_rows(dg);
                    binop_with_scalar_tile_init();
                    scalar_rows<MUL_UNARY>(dg, inv_alpha);
                    mul_binary_tile_init();
                    mul_rows(dg, du, dg);
                    binop_with_scalar_tile_init();
                    scalar_rows<MUL_UNARY>(dg, scores[e]);
                }
                tile_regs_commit();
                cb_reserve_back(cb_out, g);
                tile_regs_wait();
                for (uint32_t i = 0; i < g; ++i) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, g);
                custom_mm_block_init_short<false, kLayout>(cb_in0, cb_w, cb_out, g);
            }
        } else {
            tile_regs_acquire();
            for (uint32_t e = 0; e < num_sel; ++e) {
                for (uint32_t b = 0; b < blocks; ++b) {
                    const uint32_t cbw = block_cb(cb_w, b);
                    cb_wait_front(cbw, slot_tiles);
                    const bool bias = b + 1 == blocks;
                    const uint32_t in0_idx = e * seg + (bias ? kt : b * kb);
                    if (e + 1 == num_sel && bias) {
                        mm_block<true>(cb_in0, cbw, in0_idx, 0, 1, g);
                    } else {
                        mm_block<false>(cb_in0, cbw, in0_idx, 0, bias ? 1 : kb, g);
                    }
                    cb_pop_front(cbw, slot_tiles);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, g);
            tile_regs_wait();
            for (uint32_t i = 0; i < g; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, g);
        }
    }
    cb_pop_front(cb_in0, in0_tiles);
    custom_mm_block_uninit<kLayout>(cb_out);
}
