// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN fused verify v2, compute (fused_step/gdn_verify_v2.py), core (h, j) = value head h, value-column tile j.
// The v2 decode step's compute body (gdn_step_v2_compute.cpp, steps A-E below, same ops in the same order) in a loop
// over the T verify rows. Changes, none of which touches the arithmetic: the taps, constants, gate constants and norm
// weight are waited for once and popped at the end; token t > 0 reads its state from cb_Sn, which token t - 1's
// rank-1 update packs (fp32) next to cb_snew / cb_state (v2: the same fp32 tiles round-trip through rec_state).
//   A. dense conv: C = silu(H1*T0 + H2*T1 + H3*T2 + X*T3) on ONE tile (slot rows: q_t, k_t, v_j, z_j; z's taps
//      are 0, 0, 0, 1 so its row is silu(z))
//   B. dense L2 norms: N = C * (rsqrt(Gm @ (C^2 @ ones) + eps) * Mqk + Mvz); Gm sums the q rows (x Dk, i.e. the
//      q scale Dk^-0.5 folded in) and the k rows; v and z rows pass through (x 1). The reader splits N into row-0
//      tiles q_t, k_t, v_j, silu(z_j) and the column tiles k_t^T.
//   C. decay = exp(negA * softplus(a + dt_bias)), beta = sigmoid(b)   (v1 verbatim)
//   D. the recurrence on the Kt state tiles of value-column block j (v1's math on one column block):
//      Sd = S * decay, vread = k^T Sd, u = (v - vread) beta, Snew = Sd + k u^T, o = q^T Snew
//   E. psum = sum(o_j^2) -> writer (exchange among the head's Vt cores) -> gather -> r = rsqrt(sum / Dv + eps);
//      out = o_j * r * norm_w_j * silu(z_j)   (bf16)
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

// CB ids (gdn_step_v2.py CB)
constexpr uint32_t cb_h = 0, cb_x = 1, cb_tap = 2, cb_out = 5;
constexpr uint32_t cb_const = 6, cb_c = 7, cb_t1 = 8, cb_t2 = 9, cb_n = 10, cb_aq = 11, cb_ak = 12, cb_kcol = 13;
constexpr uint32_t cb_v = 14, cb_sz = 15, cb_a = 16, cb_b = 17, cb_na = 18, cb_dt = 19, cb_decay = 20, cb_beta = 21;
constexpr uint32_t cb_S = 22, cb_sd = 23, cb_vread = 24, cb_u = 25, cb_snew = 26, cb_state = 27, cb_o = 28;
constexpr uint32_t cb_sq = 29, cb_ps = 30, cb_gather = 31, cb_r = 32, cb_w = 33, cb_Sn = 38;
// const tiles
constexpr uint32_t C_ONES = 0, C_MCOL = 1, C_ONESM = 2, C_G = 3, C_MQK = 4, C_MVZ = 5, NCONST = 6;

constexpr uint32_t DST_TILES = 4;  // fp32 DST, half sync

void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(Mt * Nt);
    matmul_init(a, b, 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(a, b, mi * Kt + ki, ki * Nt + ni, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_o_.push_back(Mt * Nt);
}

// Snew[t] = sd[t] + kcol[t] @ u, t < Kt (one value-column block); packed to o and cb_state (and cb_Sn, the next
// token's state, when carry)
template <uint32_t Kt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t u, uint32_t o, bool carry) {
    CircularBuffer cb_o_(o);
    CircularBuffer cb_st(cb_state);
    CircularBuffer cb_sn(cb_Sn);
    cb_o_.reserve_back(Kt);
    cb_st.reserve_back(Kt);
    if (carry) {
        cb_sn.reserve_back(Kt);
    }
    for (uint32_t t0 = 0; t0 < Kt; t0 += DST_TILES) {
        const uint32_t nn = (Kt - t0 < DST_TILES) ? (Kt - t0) : DST_TILES;
        tile_regs_acquire();
        copy_init(sd);
        for (uint32_t q = 0; q < nn; q++) {
            copy_tile(sd, t0 + q, q);
        }
        matmul_init(kcol, u, 0);
        for (uint32_t q = 0; q < nn; q++) {
            matmul_tiles(kcol, u, t0 + q, 0, q);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t q = 0; q < nn; q++) {
            pack_tile<true>(q, o, t0 + q);
            pack_tile<true>(q, cb_state, t0 + q);
            if (carry) {
                pack_tile<true>(q, cb_Sn, t0 + q);
            }
        }
        tile_regs_release();
    }
    cb_o_.push_back(Kt);
    cb_st.push_back(Kt);
    if (carry) {
        cb_sn.push_back(Kt);
    }
}

void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    mul_bcast_scalar_init(a, scal);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; i++) {
        mul_tiles_bcast_scalar(a, scal, i, 0, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < n; i++) {
        pack_tile(i, o, i);
    }
    tile_regs_release();
    cb_o_.push_back(n);
}

// out <- reduce_row(in)[r] (column 0), times pre_mul, + eps, rsqrt, times the [0,0] mask when rsqrt
template <bool do_rsqrt>
void row_reduce(uint32_t in, uint32_t out, uint32_t pre_mul, uint32_t eps) {
    cb_wait_front(in, 1);
    compute_kernel_hw_startup(cb_const, in, out);  // REDUCE_ROW swaps operands: scaler -> SrcA
    cb_reserve_back(out, 1);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(in, cb_const, out);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(in, cb_const, 0, C_ONES, 0);
    reduce_uninit(in);
    if constexpr (do_rsqrt) {
        binop_with_scalar_tile_init();
        if (pre_mul != 0x3f800000u) {
            mul_unary_tile(0, pre_mul);
        }
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        copy_init(cb_const);
        copy_tile(cb_const, C_MCOL, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();
    cb_push_back(out, 1);
    cb_pop_front(in, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t l2_eps = get_compile_time_arg_val(1);
    constexpr uint32_t inv_dv = get_compile_time_arg_val(2);
    constexpr uint32_t rms_eps = get_compile_time_arg_val(3);
    constexpr uint32_t sp_beta = get_compile_time_arg_val(4);
    constexpr uint32_t sp_thr = get_compile_time_arg_val(5);
    constexpr uint32_t T = get_compile_time_arg_val(6);
    static_assert(Kt <= DST_TILES, "head dims up to 128");

    for (uint32_t tok = 0; tok < T; tok++) {
        const bool carry = tok + 1 < T;
        const uint32_t cb_Sin = tok == 0 ? cb_S : cb_Sn;

        // ---------------- A. dense conv + silu ----------------
        {
            DeviceZoneScopedN("C2-A-conv");
            cb_wait_front(cb_h, 3);
            cb_wait_front(cb_x, 1);
            cb_wait_front(cb_tap, 4);
            compute_kernel_hw_startup(cb_h, cb_tap, cb_c);
            cb_reserve_back(cb_c, 1);
            tile_regs_acquire();
            mul_init(cb_h, cb_tap);
            mul_tiles(cb_h, cb_tap, 0, 0, 0);
            mul_tiles(cb_h, cb_tap, 1, 1, 1);
            mul_tiles(cb_h, cb_tap, 2, 2, 2);
            mul_init(cb_x, cb_tap);
            mul_tiles(cb_x, cb_tap, 0, 3, 3);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(2, 3, 2);
            add_binary_tile(0, 2, 0);
            silu_tile_init();
            silu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_c);
            tile_regs_release();
            cb_push_back(cb_c, 1);
            cb_pop_front(cb_h, 3);
            cb_pop_front(cb_x, 1);
        }

        // ---------------- B. dense L2 norms ----------------
        {
            DeviceZoneScopedN("C2-B-l2");
            cb_wait_front(cb_c, 1);
            cb_wait_front(cb_const, NCONST);
            compute_kernel_hw_startup(cb_c, cb_c, cb_t1);
            // t1 = C^2
            cb_reserve_back(cb_t1, 1);
            tile_regs_acquire();
            mul_init(cb_c, cb_c);
            mul_tiles(cb_c, cb_c, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_t1);
            tile_regs_release();
            cb_push_back(cb_t1, 1);
            // t2 = t1 @ ones (row sums, every column)
            cb_wait_front(cb_t1, 1);
            cb_reserve_back(cb_t2, 1);
            tile_regs_acquire();
            matmul_init(cb_t1, cb_const, 0);
            matmul_tiles(cb_t1, cb_const, 0, C_ONESM, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_t2);
            tile_regs_release();
            cb_push_back(cb_t2, 1);
            cb_pop_front(cb_t1, 1);
            // N = C * (rsqrt(G @ t2 + eps) * Mqk + Mvz)
            cb_wait_front(cb_t2, 1);
            cb_reserve_back(cb_n, 1);
            tile_regs_acquire();
            matmul_init(cb_const, cb_t2, 0);
            matmul_tiles(cb_const, cb_t2, C_G, 0, 0);
            binop_with_scalar_tile_init();
            add_unary_tile(0, l2_eps);
            rsqrt_tile_init();
            rsqrt_tile(0);
            copy_init(cb_const);
            copy_tile(cb_const, C_MQK, 1);
            copy_tile(cb_const, C_MVZ, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
            copy_init(cb_c);
            copy_tile(cb_c, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_n);
            tile_regs_release();
            cb_push_back(cb_n, 1);
            cb_pop_front(cb_t2, 1);
            cb_pop_front(cb_c, 1);
        }

        // ---------------- C. gates (v1) ----------------
        {
            DeviceZoneScopedN("C2-C-gates");
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, 1);
            cb_wait_front(cb_na, 1);
            cb_wait_front(cb_dt, 1);
            cb_reserve_back(cb_decay, 1);
            tile_regs_acquire();
            copy_init(cb_a);
            copy_tile(cb_a, 0, 0);
            copy_init(cb_dt);
            copy_tile(cb_dt, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            softplus_tile_init();
            softplus_tile(0, sp_beta, sp_beta, sp_thr);
            copy_init(cb_na);
            copy_tile(cb_na, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            exp_tile_init();
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_decay);
            tile_regs_release();
            cb_push_back(cb_decay, 1);
            cb_reserve_back(cb_beta, 1);
            tile_regs_acquire();
            copy_init(cb_b);
            copy_tile(cb_b, 0, 0);
            sigmoid_tile_init();
            sigmoid_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_beta);
            tile_regs_release();
            cb_push_back(cb_beta, 1);
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_b, 1);
        }

        // ---------------- D. recurrence on value-column block j ----------------
        {
            DeviceZoneScopedN("C2-D-rec");
            cb_wait_front(cb_decay, 1);
            cb_wait_front(cb_Sin, Kt);
            bcast_scalar_mul(cb_Sin, cb_decay, cb_sd, Kt);
            cb_pop_front(cb_decay, 1);
            cb_pop_front(cb_Sin, Kt);
            cb_wait_front(cb_sd, Kt);
            cb_wait_front(cb_ak, Kt);
            mm(cb_ak, cb_sd, cb_vread, 1, Kt, 1);
            cb_pop_front(cb_ak, Kt);
            cb_wait_front(cb_vread, 1);
            cb_wait_front(cb_v, 1);
            cb_wait_front(cb_beta, 1);
            // u = (v - vread) * beta   (beta at [0, 0], column broadcast)
            cb_reserve_back(cb_u, 1);
            tile_regs_acquire();
            sub_init(cb_v, cb_vread);
            sub_tiles(cb_v, cb_vread, 0, 0, 0);
            copy_init(cb_beta);
            copy_tile(cb_beta, 0, 1);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_u);
            tile_regs_release();
            cb_push_back(cb_u, 1);
            cb_pop_front(cb_v, 1);
            cb_pop_front(cb_vread, 1);
            cb_pop_front(cb_beta, 1);
            cb_wait_front(cb_u, 1);
            cb_wait_front(cb_kcol, Kt);
            rank1_update<Kt>(cb_sd, cb_kcol, cb_u, cb_snew, carry);
            cb_pop_front(cb_kcol, Kt);
            cb_pop_front(cb_u, 1);
            cb_pop_front(cb_sd, Kt);
            cb_wait_front(cb_snew, Kt);
            cb_wait_front(cb_aq, Kt);
            mm(cb_aq, cb_snew, cb_o, 1, Kt, 1);
            cb_pop_front(cb_aq, Kt);
            cb_pop_front(cb_snew, Kt);
        }

        // ---------------- E. gated RMSNorm (partial sums exchanged by the writer) ----------------
        {
            DeviceZoneScopedN("C2-E-norm");
            cb_wait_front(cb_o, 1);
            compute_kernel_hw_startup(cb_o, cb_o, cb_sq);
            cb_reserve_back(cb_sq, 1);
            tile_regs_acquire();
            mul_init(cb_o, cb_o);
            mul_tiles(cb_o, cb_o, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sq);
            tile_regs_release();
            cb_push_back(cb_sq, 1);
            row_reduce<false>(cb_sq, cb_ps, 0x3f800000u, 0);
        }
        {
            DeviceZoneScopedN("C2-E-out");
            row_reduce<true>(cb_gather, cb_r, inv_dv, rms_eps);
            cb_wait_front(cb_r, 1);
            cb_wait_front(cb_w, 1);
            cb_wait_front(cb_sz, 1);
            compute_kernel_hw_startup(cb_o, cb_r, cb_out);
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            mul_bcast_cols_init(cb_o, cb_r);
            mul_tiles_bcast_cols(cb_o, cb_r, 0, 0, 0);
            copy_init(cb_w);
            copy_tile(cb_w, 0, 1);
            copy_init(cb_sz);
            copy_tile(cb_sz, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_o, 1);
            cb_pop_front(cb_r, 1);
            cb_pop_front(cb_sz, 1);
        }
    }  // token loop
    cb_pop_front(cb_tap, 4);
    cb_pop_front(cb_const, NCONST);
    cb_pop_front(cb_na, 1);
    cb_pop_front(cb_dt, 1);
    cb_pop_front(cb_w, 1);
}
