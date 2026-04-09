// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Multi-token fused GDN recurrence compute kernel with ping-pong state CBs.
//
// Per pair: state read once, then num_tokens recurrence steps with ping-pong.
// After all tokens, final state is copied to cb_state_out for the writer.
//
// Recurrence per token (identical math to gdn_recurrence.cpp):
//   1. exp(g) -> cb_exp_g
//   2. state_src * exp(g) -> state_tmp (decay)
//   3. kv_mem = k_row @ state_tmp
//   4. delta = v - kv_mem; delta_s = beta * delta
//   5. state_dst = state_tmp + outer(k_col, delta_s)
//   6. output = q @ state_dst

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/bcast.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(2);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(3);
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_row = tt::CBIndex::c_1;
    constexpr uint32_t cb_k_col = tt::CBIndex::c_2;
    constexpr uint32_t cb_v = tt::CBIndex::c_3;
    constexpr uint32_t cb_g = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta = tt::CBIndex::c_5;
    constexpr uint32_t cb_state_a = tt::CBIndex::c_6;    // ping
    constexpr uint32_t cb_state_b = tt::CBIndex::c_7;    // pong
    constexpr uint32_t cb_state_tmp = tt::CBIndex::c_8;  // decay scratch
    constexpr uint32_t cb_state_out = tt::CBIndex::c_9;  // final state for writer
    constexpr uint32_t cb_state_in = tt::CBIndex::c_10;  // initial state from reader
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_exp_g = tt::CBIndex::c_24;    // 1 tile
    constexpr uint32_t cb_kv_mem = tt::CBIndex::c_25;   // Vt tiles
    constexpr uint32_t cb_delta = tt::CBIndex::c_26;    // Vt tiles: v - kv_mem
    constexpr uint32_t cb_delta_s = tt::CBIndex::c_27;  // Vt tiles: beta * delta

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        // Wait for initial state from reader (in separate CB to avoid aliasing)
        cb_wait_front(cb_state_in, state_tiles);

        // Copy initial state from cb_state_in -> cb_state_a
        cb_reserve_back(cb_state_a, state_tiles);
        copy_tile_to_dst_init_short(cb_state_in);
        for (uint32_t s = 0; s < state_tiles; s++) {
            tile_regs_acquire();
            copy_tile(cb_state_in, s, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_a, s);
            tile_regs_release();
        }
        cb_push_back(cb_state_a, state_tiles);
        cb_pop_front(cb_state_in, state_tiles);  // Free for next pair

        for (uint32_t t = 0; t < num_tokens; t++) {
            // Ping-pong: determine source and destination CBs
            uint32_t cb_src = (t % 2 == 0) ? cb_state_a : cb_state_b;
            uint32_t cb_dst = (t % 2 == 0) ? cb_state_b : cb_state_a;

            // ---- Wait for per-token inputs ----
            cb_wait_front(cb_q, Kt);
            cb_wait_front(cb_k_row, Kt);
            cb_wait_front(cb_k_col, Kt);
            cb_wait_front(cb_v, Vt);
            cb_wait_front(cb_g, 1);
            cb_wait_front(cb_beta, 1);

            // ---- Step 1: exp(g) -> cb_exp_g ----
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

            // ---- Step 2: cb_src * exp(g) -> cb_state_tmp (decay) ----
            cb_wait_front(cb_exp_g, 1);
            cb_reserve_back(cb_state_tmp, state_tiles);
            mul_tiles_bcast_scalar_init_short(cb_src, cb_exp_g);
            for (uint32_t s = 0; s < state_tiles; s++) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_src, cb_exp_g, s, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_state_tmp, s);
                tile_regs_release();
            }
            cb_push_back(cb_state_tmp, state_tiles);
            cb_pop_front(cb_src, state_tiles);
            cb_pop_front(cb_exp_g, 1);

            // ---- Step 3: kv_mem = k_row @ cb_state_tmp ----
            cb_wait_front(cb_state_tmp, state_tiles);
            cb_reserve_back(cb_kv_mem, Vt);
            mm_init(cb_k_row, cb_state_tmp, cb_kv_mem);
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(cb_k_row, cb_state_tmp, kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_kv_mem, vt);
                tile_regs_release();
            }
            cb_push_back(cb_kv_mem, Vt);

            // ---- Step 4a: delta = v - kv_mem -> cb_delta ----
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

            // ---- Step 4b: delta_s = beta * delta -> cb_delta_s ----
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

            // ---- Step 5: cb_dst = cb_state_tmp + outer(k_col, delta_s) ----
            cb_wait_front(cb_delta_s, Vt);
            cb_reserve_back(cb_dst, state_tiles);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                for (uint32_t vt = 0; vt < Vt; vt++) {
                    uint32_t sidx = kt * Vt + vt;

                    tile_regs_acquire();
                    // Preload state_tmp[sidx] into DST[0]
                    copy_tile_to_dst_init_short(cb_state_tmp);
                    copy_tile(cb_state_tmp, sidx, 0);

                    // Accumulate outer product: DST[0] += k_col[kt] * delta_s[vt]
                    mm_init_short(cb_k_col, cb_delta_s);
                    matmul_tiles(cb_k_col, cb_delta_s, kt, vt, 0);

                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_dst, sidx);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_dst, state_tiles);
            cb_pop_front(cb_state_tmp, state_tiles);
            cb_pop_front(cb_delta_s, Vt);

            // ---- Step 6: output = q @ cb_dst ----
            cb_wait_front(cb_dst, state_tiles);
            cb_reserve_back(cb_out, Vt);
            mm_init(cb_q, cb_dst, cb_out);
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(cb_q, cb_dst, kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, vt);
                tile_regs_release();
            }
            cb_push_back(cb_out, Vt);
            // DON'T pop cb_dst — it becomes the input state for the next token

            // Pop per-token inputs
            cb_pop_front(cb_q, Kt);
            cb_pop_front(cb_k_row, Kt);
            cb_pop_front(cb_k_col, Kt);
            cb_pop_front(cb_v, Vt);
            cb_pop_front(cb_g, 1);
            cb_pop_front(cb_beta, 1);
        }

        // ---- After all tokens: copy final state to cb_state_out ----
        // Determine which CB has the final state
        uint32_t cb_final = ((num_tokens - 1) % 2 == 0) ? cb_state_b : cb_state_a;
        // Wait is already satisfied (we didn't pop it after step 6)
        cb_reserve_back(cb_state_out, state_tiles);
        copy_tile_to_dst_init_short(cb_final);
        for (uint32_t s = 0; s < state_tiles; s++) {
            tile_regs_acquire();
            copy_tile(cb_final, s, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_out, s);
            tile_regs_release();
        }
        cb_push_back(cb_state_out, state_tiles);
        cb_pop_front(cb_final, state_tiles);
    }
}
