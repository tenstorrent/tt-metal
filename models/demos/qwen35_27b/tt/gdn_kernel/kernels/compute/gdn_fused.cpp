// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused GDN compute kernel: L2 norm + gates + recurrence + RMS norm + SiLU gate.
//
// Per pair:
//   1. L2 norm Q (from raw conv_out head) + scale
//   2. L2 norm K
//   3. K transpose (for outer product)
//   4. Gates: beta = sigmoid(b), g = neg_exp_A * softplus(a + dt_bias)
//   5. Recurrence (same math as gdn_recurrence.cpp)
//   6. RMS norm on recurrence output
//   7. SiLU gate with z projection

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/bcast.h"
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/reduce.h"
#include "api/compute/transpose_wh.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);         // key dim tiles (4)
    constexpr uint32_t Vt = get_compile_time_arg_val(1);         // value dim tiles (4)
    constexpr uint32_t num_pairs = get_compile_time_arg_val(2);  // pairs on this core
    constexpr uint32_t state_tiles = Kt * Vt;                    // 16

    // --- CB assignments ---
    // Reader-filled inputs (per pair)
    constexpr uint32_t cb_q_raw = tt::CBIndex::c_0;  // [Kt] raw Q head from conv_out
    constexpr uint32_t cb_k_raw = tt::CBIndex::c_1;  // [Kt] raw K head from conv_out
    constexpr uint32_t cb_v = tt::CBIndex::c_3;      // [Vt] V head from conv_out
    constexpr uint32_t cb_a = tt::CBIndex::c_9;      // [1] gate input a
    constexpr uint32_t cb_b = tt::CBIndex::c_10;     // [1] gate input b
    // cb_z (c_11) removed — z handled by Python POST via ttnn.silu()
    constexpr uint32_t cb_neg_exp_A = tt::CBIndex::c_12;  // [1] -exp(A) constant
    constexpr uint32_t cb_dt_bias = tt::CBIndex::c_13;    // [1] dt_bias constant
    constexpr uint32_t cb_state_in = tt::CBIndex::c_6;    // [state_tiles] recurrence state

    // Reader-filled constants (once, persistent across pairs)
    constexpr uint32_t cb_norm_w = tt::CBIndex::c_14;         // [Vt] RMS norm weight
    constexpr uint32_t cb_scale = tt::CBIndex::c_15;          // [1] Q scale = Dk^(-0.5)
    constexpr uint32_t cb_rms_scale = tt::CBIndex::c_31;      // [1] sqrt(Dv) for RMS norm
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_19;  // [1] all-ones tile for reduce
    constexpr uint32_t cb_rms_eps = tt::CBIndex::c_20;        // [1] Dv*eps for RMS norm stability

    // Compute intermediates / outputs
    constexpr uint32_t cb_q = tt::CBIndex::c_17;         // [Kt] L2-normed + scaled Q
    constexpr uint32_t cb_k_row = tt::CBIndex::c_18;     // [Kt] L2-normed K
    constexpr uint32_t cb_k_col = tt::CBIndex::c_2;      // [Kt] K transposed
    constexpr uint32_t cb_g = tt::CBIndex::c_4;          // [1] computed decay gate
    constexpr uint32_t cb_beta = tt::CBIndex::c_5;       // [1] sigmoid(b)
    constexpr uint32_t cb_state_b = tt::CBIndex::c_7;    // [state_tiles] decayed state
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;  // [state_tiles] updated state → writer
    constexpr uint32_t cb_out = tt::CBIndex::c_16;       // [Vt] final output → writer
    constexpr uint32_t cb_exp_g = tt::CBIndex::c_24;     // [1] exp(g)
    constexpr uint32_t cb_kv_mem = tt::CBIndex::c_25;    // [Vt] k_row @ state
    constexpr uint32_t cb_delta = tt::CBIndex::c_26;     // [Vt] v - kv_mem
    constexpr uint32_t cb_delta_s = tt::CBIndex::c_27;   // [Vt] beta * delta
    constexpr uint32_t cb_sq_acc = tt::CBIndex::c_28;    // [1] norm sum accumulator
    constexpr uint32_t cb_tmp = tt::CBIndex::c_29;       // [1] scratch
    // cb_rec_out (c_30) removed — rec_out written directly to cb_out

    // Wait for persistent constants (reader pushes once)
    cb_wait_front(cb_norm_w, Vt);
    cb_wait_front(cb_scale, 1);
    cb_wait_front(cb_rms_scale, 1);
    cb_wait_front(cb_reduce_scaler, 1);
    cb_wait_front(cb_rms_eps, 1);

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        // Wait for all per-pair inputs from reader
        cb_wait_front(cb_q_raw, Kt);
        cb_wait_front(cb_k_raw, Kt);
        cb_wait_front(cb_v, Vt);
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_wait_front(cb_neg_exp_A, 1);
        cb_wait_front(cb_dt_bias, 1);
        cb_wait_front(cb_state_in, state_tiles);

        // ================================================================
        // Phase 1: L2 Norm Q → cb_q
        // q = q_raw / sqrt(sum(q_raw^2)) * scale
        // Using dot product: sum(x^2) = x @ x^T (avoids reduce_row)
        // ================================================================
        {
            // Step 1: Transpose q_raw → cb_sq_acc [Kt tiles]
            cb_reserve_back(cb_sq_acc, Kt);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                tile_regs_acquire();
                transpose_wh_init_short(cb_q_raw);
                transpose_wh_tile(cb_q_raw, kt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sq_acc, kt);
                tile_regs_release();
            }
            cb_push_back(cb_sq_acc, Kt);

            // Step 2: sum_sq = q_raw @ q_raw^T via matmul [1,Dk] x [Dk,1] = [1,1]
            cb_wait_front(cb_sq_acc, Kt);
            cb_reserve_back(cb_tmp, 1);
            mm_init(cb_q_raw, cb_sq_acc, cb_tmp);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                matmul_tiles(cb_q_raw, cb_sq_acc, kt, kt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);
            cb_pop_front(cb_sq_acc, Kt);

            // Step 3: rsqrt(sum_sq)
            cb_wait_front(cb_tmp, 1);
            tile_regs_acquire();
            copy_tile_init(cb_tmp);
            copy_tile(cb_tmp, 0, 0);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(cb_tmp, 1);
            cb_reserve_back(cb_tmp, 1);
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            // Step 4: Multiply rsqrt by scale → combined factor
            cb_wait_front(cb_tmp, 1);
            tile_regs_acquire();
            mul_tiles_bcast_scalar_init_short(cb_tmp, cb_scale);
            mul_tiles_bcast_scalar(cb_tmp, cb_scale, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(cb_tmp, 1);
            cb_reserve_back(cb_tmp, 1);
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            // Step 5: Normalize Q: q[kt] = q_raw[kt] * (inv_norm * scale)
            cb_wait_front(cb_tmp, 1);
            cb_reserve_back(cb_q, Kt);
            mul_tiles_bcast_scalar_init_short(cb_q_raw, cb_tmp);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_q_raw, cb_tmp, kt, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_q, kt);
                tile_regs_release();
            }
            cb_push_back(cb_q, Kt);
            cb_pop_front(cb_tmp, 1);
            cb_pop_front(cb_q_raw, Kt);
        }

        // ================================================================
        // Phase 2: L2 Norm K → cb_k_row
        // k = k_raw / sqrt(sum(k_raw^2))
        // Using dot product: sum(x^2) = x @ x^T
        // ================================================================
        {
            // Step 1: Transpose k_raw → cb_sq_acc [Kt tiles]
            cb_reserve_back(cb_sq_acc, Kt);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                tile_regs_acquire();
                transpose_wh_init_short(cb_k_raw);
                transpose_wh_tile(cb_k_raw, kt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sq_acc, kt);
                tile_regs_release();
            }
            cb_push_back(cb_sq_acc, Kt);

            // Step 2: sum_sq = k_raw @ k_raw^T via matmul [1,Dk] x [Dk,1] = [1,1]
            cb_wait_front(cb_sq_acc, Kt);
            cb_reserve_back(cb_tmp, 1);
            mm_init(cb_k_raw, cb_sq_acc, cb_tmp);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                matmul_tiles(cb_k_raw, cb_sq_acc, kt, kt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);
            cb_pop_front(cb_sq_acc, Kt);

            // Step 3: rsqrt(sum_sq)
            cb_wait_front(cb_tmp, 1);
            tile_regs_acquire();
            copy_tile_init(cb_tmp);
            copy_tile(cb_tmp, 0, 0);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(cb_tmp, 1);
            cb_reserve_back(cb_tmp, 1);
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            // Step 4: Normalize K: k[kt] = k_raw[kt] * rsqrt(sum_sq)
            cb_wait_front(cb_tmp, 1);
            cb_reserve_back(cb_k_row, Kt);
            mul_tiles_bcast_scalar_init_short(cb_k_raw, cb_tmp);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_k_raw, cb_tmp, kt, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_k_row, kt);
                tile_regs_release();
            }
            cb_push_back(cb_k_row, Kt);
            cb_pop_front(cb_tmp, 1);
            cb_pop_front(cb_k_raw, Kt);
        }

        // ================================================================
        // Phase 3: K transpose → cb_k_col
        // ================================================================

        cb_wait_front(cb_k_row, Kt);
        cb_reserve_back(cb_k_col, Kt);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            tile_regs_acquire();
            transpose_wh_init_short(cb_k_row);
            transpose_wh_tile(cb_k_row, kt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_k_col, kt);
            tile_regs_release();
        }
        cb_push_back(cb_k_col, Kt);

        // ================================================================
        // Phase 4: Gates
        // beta = sigmoid(b)
        // g = neg_exp_A * softplus(a + dt_bias)
        // ================================================================

        // beta = sigmoid(b) → cb_beta
        cb_reserve_back(cb_beta, 1);
        tile_regs_acquire();
        copy_tile_init(cb_b);
        copy_tile(cb_b, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_beta);
        tile_regs_release();
        cb_push_back(cb_beta, 1);
        cb_pop_front(cb_b, 1);

        // softplus(a + dt_bias) = log(1 + exp(a + dt_bias)) → cb_g
        cb_reserve_back(cb_g, 1);
        tile_regs_acquire();
        add_tiles_init(cb_a, cb_dt_bias);
        add_tiles(cb_a, cb_dt_bias, 0, 0, 0);  // DST[0] = a + dt_bias
        exp_tile_init();
        exp_tile(0);  // DST[0] = exp(a + dt_bias)
        log1p_tile_init();
        log1p_tile(0);  // DST[0] = log(1 + exp(a + dt_bias))
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_g);
        tile_regs_release();
        cb_push_back(cb_g, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_dt_bias, 1);

        // g = neg_exp_A * softplus_result → cb_g (in-place)
        cb_wait_front(cb_g, 1);
        tile_regs_acquire();
        mul_tiles_init(cb_g, cb_neg_exp_A);
        mul_tiles(cb_g, cb_neg_exp_A, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_g, 1);
        cb_reserve_back(cb_g, 1);
        pack_tile(0, cb_g);
        tile_regs_release();
        cb_push_back(cb_g, 1);
        cb_pop_front(cb_neg_exp_A, 1);

        // ================================================================
        // Phase 5: Recurrence (identical math to gdn_recurrence.cpp)
        // ================================================================

        // 5.1: exp(g) → cb_exp_g
        cb_wait_front(cb_g, 1);
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

        // 5.2: state_b = state_in * exp(g)
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

        // 5.3: kv_mem = k_row @ state_b → [Vt tiles]
        cb_wait_front(cb_state_b, state_tiles);
        cb_reserve_back(cb_kv_mem, Vt);
        mm_init(cb_k_row, cb_state_b, cb_kv_mem);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                matmul_tiles(cb_k_row, cb_state_b, kt, kt * Vt + vt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_kv_mem, vt);
            tile_regs_release();
        }
        cb_push_back(cb_kv_mem, Vt);

        // 5.4a: delta = v - kv_mem
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

        // 5.4b: delta_s = beta * delta
        cb_wait_front(cb_beta, 1);
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

        // 5.5: state_out = state_b + outer(k_col, delta_s)
        cb_wait_front(cb_delta_s, Vt);
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

        // 5.6: rec_out = q @ state_out → [Vt tiles] → directly to cb_out
        cb_wait_front(cb_state_out, state_tiles);
        cb_reserve_back(cb_out, Vt);
        mm_init(cb_q, cb_state_out, cb_out);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                matmul_tiles(cb_q, cb_state_out, kt, kt * Vt + vt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out, vt);
            tile_regs_release();
        }
        cb_push_back(cb_out, Vt);

        // Pop recurrence inputs (state_out stays for writer)
        cb_pop_front(cb_q, Kt);
        cb_pop_front(cb_k_row, Kt);
        cb_pop_front(cb_k_col, Kt);
        cb_pop_front(cb_v, Vt);
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_beta, 1);

        // cb_out [Vt tiles] and cb_state_out [state_tiles] ready for writer
    }

    // Pop persistent constants
    cb_pop_front(cb_norm_w, Vt);
    cb_pop_front(cb_scale, 1);
    cb_pop_front(cb_rms_scale, 1);
    cb_pop_front(cb_reduce_scaler, 1);
    cb_pop_front(cb_rms_eps, 1);
}
