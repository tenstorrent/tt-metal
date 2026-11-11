// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident slice compute: this slice's columns of the residual stream, as V 32x32 views (row r of every view
// holds elements of activation row r % R: R x 32 tiles laid end to end; R = 32: the tiles themselves). Per all-reduce
// round r = 1 .. rounds:
//   y = sum of the chips' slots (fp32 in DEST, then bf16)
//   x = (x + w_post[r] * rmsnorm(y)) [* layer_scalar after the MLP half]   (Gemma4 sandwich norm + residual)
//   h = rmsnorm(x) [* w_pre]  (the next pre-norm: with folded weights its weight is folded into the next projection,
//                    the final norm's into the LM head; with raw weights (pre_weighted) it is applied here, as the
//                    ttnn path's rms_norm(x, w))
// and the seed round: x = x0, h = rmsnorm(x). The per-row mean square spans every slice: each slice computes its
// partial F ((sum_v X_v X_v^T) . I) ONES (every element of row q: the partial sum of squares of activation row
// q % R; I the identity, F folds rows equal mod R), the BRISC exchanges the K partials, and 1 / rms follows from
// their sum. The row-sum construction follows changh95/qwen3.8-27b-p300x2 qwen36_resident streamer_compute.cpp
// rmsnorm_rows, and the slot sum its hub_compute.cpp (Apache-2.0).
//
// Compile-time args: 0 num_chips, 1 V, 2 rounds, 3 K, 4 eps bits, 5 1 / hidden bits, 6 pre_weighted
// Runtime args: the layer scalars (rounds / 2 fp32 bit patterns)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

namespace {
constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t V = get_compile_time_arg_val(1);
constexpr uint32_t rounds = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3);
constexpr uint32_t eps_bits = get_compile_time_arg_val(4);
constexpr uint32_t inv_hidden_bits = get_compile_time_arg_val(5);
constexpr uint32_t cb_slots = 0, cb_w = 1, cb_x = 2, cb_y = 3, cb_hout = 4, cb_x0 = 5, cb_const = 6, cb_pss = 7;
constexpr uint32_t cb_exch = 8, cb_rinv = 9, cb_sum = 10, cb_wpre = 11;
constexpr bool pre_weighted = get_compile_time_arg_val(6) != 0;
constexpr uint32_t kEye = 0, kOnes = 1, kFold = 2;
constexpr uint32_t kChunk = 4;
static_assert(num_chips == 1 || num_chips % 2 == 0, "slots are summed in pairs");
static_assert(K <= 8, "the exchange tiles are summed in DEST");

// cb_pss = F ((sum_v X_v X_v^T) . I) ONES over the views of `in` (fp32)
void partial_sumsq(uint32_t in) {
    cb_wait_front(in, V);
    reconfig_full_operand<SrcOrder::Reverse>(in, in);
    matmul_block_init(in, in, 1 /* transpose */, 1, 1, 1);
    pack_reconfig_data_format(cb_sum);
    pack_init(cb_sum);
    tile_regs_acquire();
    for (uint32_t v = 0; v < V; v++) {
        matmul_block(in, in, v, v, 0, 1, 1, 1, 1);
    }
    reconfig_full_operand(cb_const, cb_const);
    copy_init(cb_const);
    copy_tile(cb_const, kEye, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_sum, 1);
    pack_tile(0, cb_sum);
    cb_push_back(cb_sum, 1);
    tile_regs_release();

    cb_wait_front(cb_sum, 1);
    reconfig_full_operand<SrcOrder::Reverse>(cb_sum, cb_const);
    matmul_block_init(cb_sum, cb_const, 0, 1, 1, 1);
    tile_regs_acquire();
    matmul_block(cb_sum, cb_const, 0, kOnes, 0, 0, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_sum, 1);
    pack_tile(0, cb_sum);
    cb_push_back(cb_sum, 1);
    tile_regs_release();

    cb_wait_front(cb_sum, 2);
    reconfig_full_operand<SrcOrder::Reverse>(cb_const, cb_sum);
    matmul_block_init(cb_const, cb_sum, 0, 1, 1, 1);
    pack_reconfig_data_format(cb_pss);
    pack_init(cb_pss);
    tile_regs_acquire();
    matmul_block(cb_const, cb_sum, kFold, 1, 0, 0, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_pss, 1);
    pack_tile(0, cb_pss);
    cb_push_back(cb_pss, 1);
    tile_regs_release();
    cb_pop_front(cb_sum, 2);
}

// cb_rinv = rsqrt(sum of the `count` partial tiles of src / hidden + eps) (fp32; src unpacks straight to DEST)
void make_rinv(uint32_t src, uint32_t count) {
    cb_wait_front(src, count);
    reconfig_full_operand(src, src);
    copy_init(src);
    tile_regs_acquire();
    for (uint32_t i = 0; i < count; i++) {
        copy_tile(src, i, i);
    }
    if (count > 1) {
        add_binary_tile_init();
        for (uint32_t i = 1; i < count; i++) {
            add_binary_tile(0, i, 0);
        }
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_hidden_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_rinv);
    pack_init(cb_rinv);
    cb_reserve_back(cb_rinv, 1);
    pack_tile(0, cb_rinv);
    cb_push_back(cb_rinv, 1);
    tile_regs_release();
    cb_pop_front(src, count);
}

void rinv_of(uint32_t in) {
    partial_sumsq(in);
    if constexpr (K == 1) {
        make_rinv(cb_pss, 1);
    } else {
        make_rinv(cb_exch, K);  // the BRISC's exchange of every slice's cb_pss
    }
}

// out = in * rinv [* w + x [* scalar]] over the views
void apply(uint32_t in, uint32_t out, bool post, uint32_t scalar) {
    cb_wait_front(cb_rinv, 1);
    cb_reserve_back(out, V);
    pack_reconfig_data_format(out);
    pack_init(out);
    for (uint32_t v0 = 0; v0 < V; v0 += kChunk) {
        const uint32_t W = V - v0 < kChunk ? V - v0 : kChunk;
        tile_regs_acquire();
        reconfig_full_operand(in, cb_rinv);
        mul_init(in, cb_rinv);
        for (uint32_t i = 0; i < W; i++) {
            mul_tiles(in, cb_rinv, v0 + i, 0, i);
        }
        if (post) {
            reconfig_full_operand(cb_w, cb_w);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
            for (uint32_t i = 0; i < W; i++) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, v0 + i, i);
            }
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x);
            for (uint32_t i = 0; i < W; i++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x, v0 + i, i);
            }
            if (scalar) {
                binop_with_scalar_tile_init();
                for (uint32_t i = 0; i < W; i++) {
                    mul_unary_tile(i, scalar);
                }
            }
        } else if (pre_weighted) {
            reconfig_full_operand(cb_wpre, cb_wpre);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_wpre);
            for (uint32_t i = 0; i < W; i++) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_wpre, v0 + i, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, out, v0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(out, V);
    cb_pop_front(cb_rinv, 1);
}

// cb_hout = rmsnorm(x) [* w_pre] (x stays at the front of cb_x)
void next_h() {
    rinv_of(cb_x);
    if constexpr (pre_weighted) {
        cb_wait_front(cb_wpre, V);
    }
    apply(cb_x, cb_hout, false, 0);
    if constexpr (pre_weighted) {
        cb_pop_front(cb_wpre, V);
    }
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_slots, cb_slots, cb_y);
    // the seed round: x = x0 (from the BRISC; the packer is cb_x's only producer)
    cb_wait_front(cb_const, 3);
    cb_wait_front(cb_x0, V);
    cb_reserve_back(cb_x, V);
    pack_reconfig_data_format(cb_x);
    pack_init(cb_x);
    for (uint32_t v0 = 0; v0 < V; v0 += kChunk) {
        const uint32_t W = V - v0 < kChunk ? V - v0 : kChunk;
        reconfig_full_operand(cb_x0, cb_x0);
        copy_init(cb_x0);
        tile_regs_acquire();
        for (uint32_t i = 0; i < W; i++) {
            copy_tile(cb_x0, v0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, cb_x, v0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_x, V);
    cb_pop_front(cb_x0, V);
    next_h();
    for (uint32_t r = 1; r <= rounds; r++) {
        // y = the sum of the chips' slots
        cb_wait_front(cb_slots, num_chips * V);
        cb_reserve_back(cb_y, V);
        pack_reconfig_data_format(cb_y);
        pack_init(cb_y);
        for (uint32_t v0 = 0; v0 < V; v0 += kChunk) {
            const uint32_t W = V - v0 < kChunk ? V - v0 : kChunk;
            reconfig_full_operand(cb_slots, cb_slots);
            tile_regs_acquire();
            for (uint32_t c = 0; c < num_chips; c += 2) {
                add_init(cb_slots, cb_slots, c > 0 /* acc_to_dest */);
                for (uint32_t i = 0; i < W; i++) {
                    add_tiles(cb_slots, cb_slots, c * V + v0 + i, (c + 1) * V + v0 + i, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < W; i++) {
                pack_tile<true>(i, cb_y, v0 + i);
            }
            tile_regs_release();
        }
        cb_push_back(cb_y, V);
        cb_pop_front(cb_slots, num_chips * V);
        // x = (x + w * rmsnorm(y)) [* scalar]
        rinv_of(cb_y);
        cb_wait_front(cb_w, V);
        cb_wait_front(cb_x, V);
        apply(cb_y, cb_x, true, r % 2 == 0 ? get_arg_val<uint32_t>(r / 2 - 1) : 0);
        cb_pop_front(cb_x, V);  // the old x (the new one follows it in the CB)
        cb_pop_front(cb_w, V);
        cb_pop_front(cb_y, V);
        next_h();
    }
}
