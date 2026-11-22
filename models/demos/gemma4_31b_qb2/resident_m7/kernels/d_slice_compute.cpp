// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter slice compute (TRISC; M6b): this slice's columns of the drafter's residual stream at R = 32
// (V 32 x 32 tiles, rows = the launch's rows: U user blocks of Mb rows). Per drafter layer (Qwen3 pre-norm with
// DFlash2's dynamic 2-tap block convs, tt/dflash2.py _layer):
//   h   = rmsnorm(x) * w_in                       -> cb_hout (multicast for the coefficient projection) and cb_hk
//   c_s = coef_s @ E + base_s, s = 0..3           (the streamers' coefficient columns expanded to this slice's channels:
//                                                  one coefficient per 16-channel group; E the 0/1 group expansion)
//   xin = h * c_0 + (S @ h) * c_1                 -> cb_hout (multicast for q|k|v); S shifts each block's rows down one
//   y   = the chips' o_proj partials summed (or, teacher-forced, the injected attention output)
//   x   = x + y * c_2 + (S @ y) * c_3
//   h2  = rmsnorm(x) * w_post, the MLP coefficients, m_in = conv_in(h2) (as above), y = the down partials summed,
//   x   = x + conv_out(y)
// then h = rmsnorm(x) * w_final for the LM head. The per-row rms is slice_compute.cpp's (the target's): each slice's
// partial F ((sum_v X_v X_v^T) . I) ONES, exchanged by the BRISC, 1 / rms from their sum.
//
// Compile-time args: 0 num_chips, 1 V, 2 layers, 3 K, 4 eps bits, 5 1 / hidden bits, 6 inject (the attention rounds'
// y from cb_inj)
// Runtime args: 0 the first channel tile of this slice (k T)
// CB 6 (cb_const) holds: 0 eye, 1 ones, 2 fold, 3 S, 4 .. 19 E_m (m = 0..15: coefficient column 2 m -> channels
// 0..15, 2 m + 1 -> 16..31). Shared buffers (one producer and one consumer RISC per CB id, every use ordered through
// this kernel's own sequence): x0 | hk | y (x0 is read before the first h is packed; hk is read by conv_in before the
// slot sum packs y; y is read before the next h packs hk) and base | c (expand writes each coefficient tile over the
// base tile it read).

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
constexpr bool inject = get_compile_time_arg_val(6) != 0;
constexpr uint32_t cb_slots = 0, cb_x = 2, cb_y = 3, cb_hout = 4, cb_x0 = 5, cb_const = 6, cb_pss = 7;
constexpr uint32_t cb_exch = 8, cb_rinv = 9, cb_sum = 10, cb_wpre = 11, cb_hk = 12, cb_coef = 13, cb_base = 14;
constexpr uint32_t cb_c = 15, cb_inj = 16;
constexpr uint32_t kEye = 0, kOnes = 1, kFold = 2, kS = 3, kE0 = 4;
constexpr uint32_t kCoefPerSection = 12;  // 384 coefficient columns per (io, tap) section
constexpr uint32_t kChunk = 4;            // tiles per DEST pass where two DEST tiles each are live (8 fp32 tiles)
static_assert(num_chips == 1 || num_chips % 2 == 0, "slots are summed in pairs");
static_assert(K <= 8, "the exchange tiles are summed in DEST");

// cb_pss = F ((sum_v X_v X_v^T) . I) ONES over the views of `in` (fp32): slice_compute.cpp's
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

// cb_rinv = rsqrt(sum of the `count` partial tiles of src / hidden + eps) (fp32)
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

// h = rmsnorm(x) * w (x stays at the front of cb_x) -> cb_hout, and cb_hk when keep (the conv's copy)
void norm_h(bool keep) {
    partial_sumsq(cb_x);
    if constexpr (K == 1) {
        make_rinv(cb_pss, 1);
    } else {
        make_rinv(cb_exch, K);  // the BRISC's exchange of every slice's cb_pss
    }
    cb_wait_front(cb_rinv, 1);
    cb_wait_front(cb_wpre, V);
    cb_reserve_back(cb_hout, V);
    if (keep) {
        cb_reserve_back(cb_hk, V);
    }
    pack_reconfig_data_format(cb_hout);
    pack_init(cb_hout);
    for (uint32_t v0 = 0; v0 < V; v0 += 8) {
        const uint32_t W = V - v0 < 8 ? V - v0 : 8;
        tile_regs_acquire();
        reconfig_full_operand(cb_x, cb_rinv);
        mul_init(cb_x, cb_rinv);
        for (uint32_t i = 0; i < W; i++) {
            mul_tiles(cb_x, cb_rinv, v0 + i, 0, i);
        }
        reconfig_full_operand(cb_wpre, cb_wpre);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_wpre);
        for (uint32_t i = 0; i < W; i++) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_wpre, v0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            pack_tile<true>(i, cb_hout, v0 + i);
            if (keep) {
                pack_tile<true>(i, cb_hk, v0 + i);
            }
        }
        tile_regs_release();
    }
    cb_push_back(cb_hout, V);
    if (keep) {
        cb_push_back(cb_hk, V);
    }
    cb_pop_front(cb_wpre, V);
    cb_pop_front(cb_rinv, 1);
}

// cb_c[s V + t] = coef[12 s + j / 16] @ E_{j % 16} + base[s V + t] (j = j0 + t: this slice's channel tile)
void expand() {
    const uint32_t j0 = get_arg_val<uint32_t>(0);
    cb_wait_front(cb_coef, 4 * kCoefPerSection);
    cb_wait_front(cb_base, 4 * V);
    cb_reserve_back(cb_c, 4 * V);
    pack_reconfig_data_format(cb_c);
    pack_init(cb_c);
    for (uint32_t s = 0; s < 4; s++) {
        for (uint32_t t0 = 0; t0 < V; t0 += 8) {
            const uint32_t W = V - t0 < 8 ? V - t0 : 8;
            tile_regs_acquire();
            reconfig_full_operand<SrcOrder::Reverse>(cb_coef, cb_const);
            matmul_block_init(cb_coef, cb_const, 0, 1, 1, 1);
            for (uint32_t i = 0; i < W; i++) {
                const uint32_t j = j0 + t0 + i;
                matmul_block(cb_coef, cb_const, s * kCoefPerSection + j / 16, kE0 + j % 16, i, 0, 1, 1, 1);
            }
            reconfig_full_operand(cb_base, cb_base);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_base);
            for (uint32_t i = 0; i < W; i++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_base, s * V + t0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < W; i++) {
                pack_tile<true>(i, cb_c, s * V + t0 + i);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_c, 4 * V);
    cb_pop_front(cb_coef, 4 * kCoefPerSection);
    cb_pop_front(cb_base, 4 * V);
}

// out[t] = in[t] * c[s0 V + t] + (S @ in[t]) * c[(s0 + 1) V + t] [+ x[t]] over the V tiles
void conv(uint32_t in, uint32_t s0, uint32_t out, bool residual) {
    cb_wait_front(in, V);
    cb_wait_front(cb_c, 4 * V);
    cb_reserve_back(out, V);
    pack_reconfig_data_format(out);
    pack_init(out);
    for (uint32_t t0 = 0; t0 < V; t0 += kChunk) {
        const uint32_t W = V - t0 < kChunk ? V - t0 : kChunk;
        tile_regs_acquire();
        // DEST[kChunk + i] = S @ in[t]
        reconfig_full_operand<SrcOrder::Reverse>(cb_const, in);
        matmul_block_init(cb_const, in, 0, 1, 1, 1);
        for (uint32_t i = 0; i < W; i++) {
            matmul_block(cb_const, in, kS, t0 + i, kChunk + i, 0, 1, 1, 1);
        }
        // DEST[kChunk + i] *= c[(s0 + 1) V + t]
        reconfig_full_operand(cb_c, cb_c);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_c);
        for (uint32_t i = 0; i < W; i++) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_c, (s0 + 1) * V + t0 + i, kChunk + i);
        }
        // DEST[i] = in[t] * c[s0 V + t]
        reconfig_full_operand(in, cb_c);
        mul_init(in, cb_c);
        for (uint32_t i = 0; i < W; i++) {
            mul_tiles(in, cb_c, t0 + i, s0 * V + t0 + i, i);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < W; i++) {
            add_binary_tile(i, kChunk + i, i);
        }
        if (residual) {
            reconfig_full_operand(cb_x, cb_x);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x);
            for (uint32_t i = 0; i < W; i++) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_x, t0 + i, i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < W; i++) {
            // the residual's new x follows the old one in cb_x (its own V-page region: no reservation straddles)
            pack_tile<true>(i, out, t0 + i);
        }
        tile_regs_release();
    }
    cb_push_back(out, V);
}

// cb_y = the sum of the chips' slots (fp32 in DEST, then bf16), as slice_compute.cpp
void sum_slots() {
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
}

// one sublayer: h -> (coefficients) -> conv_in(h) out; then x += conv_out(y of the all-reduce round)
void sublayer(bool attention) {
    norm_h(true);
    expand();
    conv(cb_hk, 0, cb_hout, false);
    cb_pop_front(cb_hk, V);
    sum_slots();
    uint32_t y = cb_y;
    if (inject && attention) {
        cb_pop_front(cb_y, V);
        y = cb_inj;
    }
    conv(y, 2, cb_x, true);
    cb_pop_front(cb_x, V);  // the old x (the new one follows it in the CB)
    cb_pop_front(y, V);
    cb_pop_front(cb_c, 4 * V);
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_slots, cb_slots, cb_y);
    // the seed: x = x0 (from the BRISC; the packer is cb_x's only producer)
    cb_wait_front(cb_const, kE0 + 16);
    cb_wait_front(cb_x0, V);
    cb_reserve_back(cb_x, V);
    pack_reconfig_data_format(cb_x);
    pack_init(cb_x);
    for (uint32_t v0 = 0; v0 < V; v0 += 8) {
        const uint32_t W = V - v0 < 8 ? V - v0 : 8;
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
    for (uint32_t l = 0; l < layers; l++) {
        sublayer(true);
        sublayer(false);
    }
    norm_h(false);  // the final norm: the LM head's input
}
