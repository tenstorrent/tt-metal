// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter ingest, slice compute (TRISC; M6b C1), see d_ingest_slice_dm.cpp:
//   y   = the chips' fc partials summed (fp32 in DEST, then bf16), as slice_compute.cpp
//   ctx = rmsnorm(y) * w_hidden (the per-row mean square over every slice's columns, exchanged) -> cb_hout
//   per drafter layer, head block k's 8 tiles (32 rows x 256):
//     k head (k < 4): kn = rmsnorm_256(k) * w_knorm[l] (one head: a local row sum, no exchange), then RoPE
//                     (rotate-half: column tile t pairs with t +- 4) out = kn cos + rot(kn) sin -> BF8
//     v head: as is -> BF8
// The BF8 pack (cb_out8, bfloat8_b) rounds once from DEST.
//
// Compile-time args: 0 num_chips, 1 V, 2 layers, 3 K, 4 eps bits, 5 1 / hidden bits, 6 1 / head_dim bits
// Runtime args: 0 is_k

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
constexpr uint32_t layers = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3);
constexpr uint32_t eps_bits = get_compile_time_arg_val(4);
constexpr uint32_t inv_hidden_bits = get_compile_time_arg_val(5);
constexpr uint32_t inv_head_bits = get_compile_time_arg_val(6);
constexpr uint32_t cb_slots = 0, cb_y = 3, cb_hout = 4, cb_const = 6, cb_pss = 7, cb_exch = 8, cb_rinv = 9;
constexpr uint32_t cb_sum = 10, cb_wpre = 11, cb_kv = 12, cb_wk = 13, cb_rope = 14, cb_kn = 15, cb_out8 = 16;
// the head norm's partial: its own CB (cb_pss's consumer is the BRISC's exchange; one consumer RISC per CB)
constexpr uint32_t cb_hss = 19;
constexpr uint32_t kEye = 0, kOnes = 1, kFold = 2;
constexpr uint32_t kHeadTiles = 8, kHalf = 4;
static_assert(num_chips == 1 || num_chips % 2 == 0, "slots are summed in pairs");
static_assert(K <= 8, "the exchange tiles are summed in DEST");

// out = F ((sum_v X_v X_v^T) . I) ONES over n tiles of `in` (fp32): slice_compute.cpp's
void partial_sumsq(uint32_t in, uint32_t n, uint32_t out) {
    cb_wait_front(in, n);
    reconfig_full_operand<SrcOrder::Reverse>(in, in);
    matmul_block_init(in, in, 1 /* transpose */, 1, 1, 1);
    pack_reconfig_data_format(cb_sum);
    pack_init(cb_sum);
    tile_regs_acquire();
    for (uint32_t v = 0; v < n; v++) {
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
    pack_reconfig_data_format(out);
    pack_init(out);
    tile_regs_acquire();
    matmul_block(cb_const, cb_sum, kFold, 1, 0, 0, 1, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(out, 1);
    pack_tile(0, out);
    cb_push_back(out, 1);
    tile_regs_release();
    cb_pop_front(cb_sum, 2);
}

// cb_rinv = rsqrt(sum of the `count` partial tiles of src * inv + eps) (fp32)
void make_rinv(uint32_t src, uint32_t count, uint32_t inv_bits) {
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
    mul_unary_tile(0, inv_bits);
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

// out = in * rinv * w over n tiles (in and w at their fronts; rinv consumed)
void apply(uint32_t in, uint32_t w, uint32_t out, uint32_t n) {
    cb_wait_front(cb_rinv, 1);
    cb_wait_front(w, n);
    cb_reserve_back(out, n);
    pack_reconfig_data_format(out);
    pack_init(out);
    for (uint32_t v0 = 0; v0 < n; v0 += 8) {
        const uint32_t W = n - v0 < 8 ? n - v0 : 8;
        tile_regs_acquire();
        reconfig_full_operand(in, cb_rinv);
        mul_init(in, cb_rinv);
        for (uint32_t i = 0; i < W; i++) {
            mul_tiles(in, cb_rinv, v0 + i, 0, i);
        }
        reconfig_full_operand(w, w);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(w);
        for (uint32_t i = 0; i < W; i++) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(w, v0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, out, v0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(out, n);
    cb_pop_front(cb_rinv, 1);
}
}  // namespace

void kernel_main() {
    const bool is_k = get_arg_val<uint32_t>(0) != 0;
    compute_kernel_hw_startup(cb_slots, cb_slots, cb_y);
    cb_wait_front(cb_const, 3);
    // y = the sum of the chips' slots
    cb_wait_front(cb_slots, num_chips * V);
    cb_reserve_back(cb_y, V);
    pack_reconfig_data_format(cb_y);
    pack_init(cb_y);
    for (uint32_t v0 = 0; v0 < V; v0 += 8) {
        const uint32_t W = V - v0 < 8 ? V - v0 : 8;
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
    // ctx = rmsnorm(y) * w_hidden
    partial_sumsq(cb_y, V, cb_pss);
    make_rinv(cb_exch, K, inv_hidden_bits);
    apply(cb_y, cb_wpre, cb_hout, V);
    cb_pop_front(cb_wpre, V);
    cb_pop_front(cb_y, V);

    if (is_k) {
        cb_wait_front(cb_rope, 2 * kHeadTiles);  // cos 0..7, sin 8..15
    }
    for (uint32_t l = 0; l < layers; l++) {
        cb_wait_front(cb_kv, kHeadTiles);
        cb_reserve_back(cb_out8, kHeadTiles);
        if (is_k) {
            partial_sumsq(cb_kv, kHeadTiles, cb_hss);
            make_rinv(cb_hss, 1, inv_head_bits);
            apply(cb_kv, cb_wk, cb_kn, kHeadTiles);
            cb_pop_front(cb_wk, kHeadTiles);
            cb_wait_front(cb_kn, kHeadTiles);
            pack_reconfig_data_format(cb_out8);
            pack_init(cb_out8);
            for (uint32_t t0 = 0; t0 < kHeadTiles; t0 += kHalf) {
                tile_regs_acquire();
                reconfig_full_operand(cb_kn, cb_rope);
                mul_init(cb_kn, cb_rope);
                for (uint32_t i = 0; i < kHalf; i++) {
                    const uint32_t t = t0 + i, partner = t < kHalf ? t + kHalf : t - kHalf;
                    mul_tiles(cb_kn, cb_rope, t, t, i);                             // kn cos
                    mul_tiles(cb_kn, cb_rope, partner, kHeadTiles + t, kHalf + i);  // kn[partner] sin
                }
                if (t0 == 0) {
                    sub_binary_tile_init();  // first half: kn cos - kn[t + 4] sin
                    for (uint32_t i = 0; i < kHalf; i++) {
                        sub_binary_tile(i, kHalf + i, i);
                    }
                } else {
                    add_binary_tile_init();  // second half: kn cos + kn[t - 4] sin
                    for (uint32_t i = 0; i < kHalf; i++) {
                        add_binary_tile(i, kHalf + i, i);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < kHalf; i++) {
                    pack_tile<true>(i, cb_out8, t0 + i);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_kn, kHeadTiles);
        } else {
            pack_reconfig_data_format(cb_out8);
            pack_init(cb_out8);
            reconfig_full_operand(cb_kv, cb_kv);
            copy_init(cb_kv);
            tile_regs_acquire();
            for (uint32_t t = 0; t < kHeadTiles; t++) {
                copy_tile(cb_kv, t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < kHeadTiles; t++) {
                pack_tile<true>(t, cb_out8, t);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out8, kHeadTiles);
        cb_pop_front(cb_kv, kHeadTiles);
    }
}
