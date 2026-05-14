// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Native-input fused GDN recurrence compute kernel.
//
// Differences vs gdn_recurrence.cpp:
//   - Q is scaled inside the kernel: cb_q_scaled = cb_q * cb_scale  (saves an upstream
//     ttnn.multiply(q, scale)).
//   - K_col is derived inside the kernel via transpose_wh on cb_k  (saves an upstream
//     ttnn.transpose and a retile).
//   - cb_scale is read once (persistent across pairs); cb_q_scaled and cb_k_col are
//     internal CBs (compute-owned).
//
// Flow per pair:
//   q_scaled = q * scale                (Phase 0a)
//   k_col    = transpose_wh(k)          (Phase 0b)
//   state_b  = state_in * exp(g)        (Steps 1-2)
//   kv_mem   = k @ state_b              (Step 3)
//   delta    = v - kv_mem               (Step 4a)
//   delta_s  = beta * delta             (Step 4b)
//   state_out= state_b + outer(k_col, delta_s)   (Step 5)
//   output   = q_scaled @ state_out     (Step 6)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(2);
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_q = tt::CBIndex::c_0;          // reader-filled: unscaled Q
    constexpr uint32_t cb_k = tt::CBIndex::c_1;          // reader-filled: K rows
    constexpr uint32_t cb_k_col = tt::CBIndex::c_2;      // compute-filled: K transposed
    constexpr uint32_t cb_v = tt::CBIndex::c_3;          // reader-filled
    constexpr uint32_t cb_g = tt::CBIndex::c_4;          // reader-filled: g scalar
    constexpr uint32_t cb_beta = tt::CBIndex::c_5;       // reader-filled: beta scalar
    constexpr uint32_t cb_state_in = tt::CBIndex::c_6;   // reader-filled
    constexpr uint32_t cb_state_b = tt::CBIndex::c_7;    // intermediate (decayed state)
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;  // compute-filled → writer reads
    constexpr uint32_t cb_scale = tt::CBIndex::c_15;     // reader-filled, persistent
    constexpr uint32_t cb_out = tt::CBIndex::c_16;       // compute-filled → writer reads
    constexpr uint32_t cb_exp_g = tt::CBIndex::c_24;     // 1 tile
    constexpr uint32_t cb_kv_mem = tt::CBIndex::c_25;    // Vt tiles
    constexpr uint32_t cb_delta = tt::CBIndex::c_26;     // Vt tiles
    constexpr uint32_t cb_delta_s = tt::CBIndex::c_27;   // Vt tiles
    constexpr uint32_t cb_q_scaled = tt::CBIndex::c_28;  // compute-internal: Q*scale, Kt tiles

    // Wait for scale (once, persistent across all pairs)
    cb_wait_front(cb_scale, 1);

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        // ---- Wait for inputs ----
        cb_wait_front(cb_state_in, state_tiles);
        cb_wait_front(cb_q, Kt);
        cb_wait_front(cb_k, Kt);
        cb_wait_front(cb_v, Vt);
        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_beta, 1);

        // ---- Phase 0a: cb_q_scaled = cb_q * cb_scale (bcast scalar) ----
        cb_reserve_back(cb_q_scaled, Kt);
        mul_tiles_bcast_scalar_init_short(cb_q, cb_scale);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_q, cb_scale, kt, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_q_scaled, kt);
            tile_regs_release();
        }
        cb_push_back(cb_q_scaled, Kt);

        // ---- Phase 0b: cb_k_col = transpose_wh(cb_k) ----
        cb_reserve_back(cb_k_col, Kt);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            tile_regs_acquire();
            transpose_wh_init_short(cb_k);
            transpose_wh_tile(cb_k, kt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_k_col, kt);
            tile_regs_release();
        }
        cb_push_back(cb_k_col, Kt);

        // ---- Step 1: exp(g) → cb_exp_g ----
        cb_reserve_back(cb_exp_g, 1);
        tile_regs_acquire();
        copy_tile_init(cb_g);
        copy_tile(cb_g, 0, 0);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_exp_g);
        tile_regs_release();
        cb_push_back(cb_exp_g, 1);

        // ---- Step 2: state_b = state_in * exp(g) ----
        cb_wait_front(cb_exp_g, 1);
        cb_reserve_back(cb_state_b, state_tiles);
        mul_tiles_bcast_scalar_init_short(cb_state_in, cb_exp_g);
        for (uint32_t s = 0; s < state_tiles; s++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_state_in, cb_exp_g, s, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_b, s);
            tile_regs_release();
        }
        cb_push_back(cb_state_b, state_tiles);
        cb_pop_front(cb_state_in, state_tiles);
        cb_pop_front(cb_exp_g, 1);

        // ---- Step 3: kv_mem = k @ state_b ----
        cb_wait_front(cb_state_b, state_tiles);
        cb_reserve_back(cb_kv_mem, Vt);
        mm_init(cb_k, cb_state_b, cb_kv_mem);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                matmul_tiles(cb_k, cb_state_b, kt, kt * Vt + vt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_kv_mem, vt);
            tile_regs_release();
        }
        cb_push_back(cb_kv_mem, Vt);

        // ---- Step 4a: delta = v - kv_mem ----
        cb_wait_front(cb_kv_mem, Vt);
        cb_reserve_back(cb_delta, Vt);
        sub_tiles_init(cb_v, cb_kv_mem);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            sub_tiles(cb_v, cb_kv_mem, vt, vt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_delta, vt);
            tile_regs_release();
        }
        cb_push_back(cb_delta, Vt);
        cb_pop_front(cb_kv_mem, Vt);

        // ---- Step 4b: delta_s = beta * delta ----
        cb_wait_front(cb_delta, Vt);
        cb_reserve_back(cb_delta_s, Vt);
        mul_tiles_bcast_scalar_init_short(cb_delta, cb_beta);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_delta, cb_beta, vt, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_delta_s, vt);
            tile_regs_release();
        }
        cb_push_back(cb_delta_s, Vt);
        cb_pop_front(cb_delta, Vt);

        // ---- Step 5: state_out = state_b + outer(k_col, delta_s) ----
        cb_wait_front(cb_delta_s, Vt);
        cb_wait_front(cb_k_col, Kt);
        cb_reserve_back(cb_state_out, state_tiles);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                uint32_t sidx = kt * Vt + vt;
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_state_b);
                copy_tile(cb_state_b, sidx, 0);
                mm_init_short(cb_k_col, cb_delta_s);
                matmul_tiles(cb_k_col, cb_delta_s, kt, vt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_state_out, sidx);
                tile_regs_release();
            }
        }
        cb_push_back(cb_state_out, state_tiles);
        cb_pop_front(cb_state_b, state_tiles);
        cb_pop_front(cb_delta_s, Vt);
        cb_pop_front(cb_k_col, Kt);

        // ---- Step 6: output = q_scaled @ state_out ----
        cb_wait_front(cb_q_scaled, Kt);
        cb_wait_front(cb_state_out, state_tiles);
        cb_reserve_back(cb_out, Vt);
        mm_init(cb_q_scaled, cb_state_out, cb_out);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                matmul_tiles(cb_q_scaled, cb_state_out, kt, kt * Vt + vt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out, vt);
            tile_regs_release();
        }
        cb_push_back(cb_out, Vt);

        // Pop per-pair inputs (state_out is popped by writer)
        cb_pop_front(cb_q, Kt);
        cb_pop_front(cb_q_scaled, Kt);
        cb_pop_front(cb_k, Kt);
        cb_pop_front(cb_v, Vt);
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_beta, 1);
    }

    cb_pop_front(cb_scale, 1);
}
