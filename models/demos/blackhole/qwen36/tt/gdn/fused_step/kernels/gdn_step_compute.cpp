// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One Gated DeltaNet DECODE step for one value head, everything between the in-projection and the out-projection
// (fused_step/gdn_step.py). Per-token vectors are TILE row vectors: the token in row 0, rows 1..31 padding.
//   A. conv: c = silu(h1*t0 + h2*t1 + h3*t2 + x*t3) over this head's q|k|v channel tiles (bf16 in, fp32 out);
//      sz = silu(z)
//   B. q = l2norm(q) * scale, k = l2norm(k): x / sqrt(sum x^2 + eps) per row; rows 1..31 forced to 0 (the rank-1
//      update below contracts over the 32 tile rows); v rows 1..31 forced to 0 as well
//   C. decay = exp(negA * softplus(a + dt_bias)), beta = sigmoid(b)   (scalars at [0,0])
//   D. the recurrence of transformer/fused_recurrent_gated_delta_rule (decay before read, rank-1 update, read q from
//      the updated state), T = 1
//   E. out = o * rsqrt(mean(o^2) + eps) * norm_w * sz   (bf16)
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
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

namespace {

// CB ids (gdn_step.py CBS)
constexpr uint32_t cb_h = 0, cb_x = 1, cb_tap = 2, cb_z = 3, cb_a = 4, cb_b = 5, cb_na = 6, cb_dt = 7, cb_w = 8;
constexpr uint32_t cb_mcol = 9, cb_one = 10, cb_S = 11, cb_c = 12, cb_sz = 13, cb_sq = 14, cb_r = 15;
constexpr uint32_t cb_qn = 16, cb_kn = 17, cb_v = 18, cb_decay = 19, cb_beta = 20, cb_sd = 21, cb_vread = 22;
constexpr uint32_t cb_u = 23, cb_kcol = 24, cb_snew = 25, cb_state = 26, cb_o = 27, cb_out = 28;

constexpr uint32_t DST_TILES = 4;  // fp32 DST, half sync

// ---- the recurrence helpers of fused_recurrent_gated_delta_rule's compute kernel (verbatim math) ----
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(Mt * Nt);
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_o_.push_back(Mt * Nt);
}

template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t u, uint32_t o) {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_o_(o);
    CircularBuffer cb_st(cb_state);
    cb_o_.reserve_back(kv);
    cb_st.reserve_back(kv);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = (Vt - n0 < DST_TILES) ? (Vt - n0) : DST_TILES;
            tile_regs_acquire();
            copy_init(sd);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(sd, mi * Vt + n0 + j, j);
            }
            matmul_init(kcol, u, 0);
            for (uint32_t j = 0; j < nn; j++) {
                matmul_tiles(kcol, u, mi, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, o, mi * Vt + n0 + j);
                pack_tile<true>(j, cb_state, mi * Vt + n0 + j);
            }
            tile_regs_release();
        }
    }
    cb_o_.push_back(kv);
    cb_st.push_back(kv);
}

void delta_rule_residual(uint32_t a, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = (n - i0 < per_batch) ? (n - i0) : per_batch;
        tile_regs_acquire();
        sub_init(a, b);
        for (uint32_t j = 0; j < nn; j++) {
            sub_tiles(a, b, i0 + j, i0 + j, j);
        }
        copy_init(beta);
        copy_tile(beta, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o_.push_back(n);
}

void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o_.push_back(n);
}

void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o_.push_back(n);
}

// sum over n (<= 4) tiles of src[base..base+n)^2 -> one tile in cb_sq (elementwise, fp32)
void sum_squares(uint32_t src, uint32_t base, uint32_t n) {
    compute_kernel_hw_startup(src, src, cb_sq);
    cb_reserve_back(cb_sq, 1);
    tile_regs_acquire();
    mul_init(src, src);
    for (uint32_t t = 0; t < n; t++) {
        mul_tiles(src, src, base + t, base + t, t);
    }
    add_binary_tile_init();
    for (uint32_t t = 1; t < n; t++) {
        add_binary_tile(0, t, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sq);
    tile_regs_release();
    cb_push_back(cb_sq, 1);
}

// cb_r <- rsqrt(rowsum(cb_sq) * pre_mul + eps) * post_mul, only [0, 0] kept (column 0, row 0): the per-row factor
// for mul_tiles_bcast_cols with the padding rows forced to zero.
void row_rsqrt(uint32_t pre_mul, uint32_t eps, uint32_t post_mul) {
    cb_wait_front(cb_sq, 1);
    compute_kernel_hw_startup(cb_one, cb_sq, cb_r);  // REDUCE_ROW swaps operands: scaler -> SrcA
    cb_reserve_back(cb_r, 1);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, cb_r);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, 0, 0, 0);
    reduce_uninit(cb_sq);
    binop_with_scalar_tile_init();
    if (pre_mul != 0x3f800000u) {
        mul_unary_tile(0, pre_mul);
    }
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if (post_mul != 0x3f800000u) {
        mul_unary_tile(0, post_mul);
    }
    copy_init(cb_mcol);
    copy_tile(cb_mcol, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    cb_push_back(cb_r, 1);
    cb_pop_front(cb_sq, 1);
}

// out[0..n) = src[base..base+n) * cb_r (column broadcast)
void scale_rows(uint32_t src, uint32_t base, uint32_t n, uint32_t out) {
    cb_wait_front(cb_r, 1);
    compute_kernel_hw_startup(src, cb_r, out);
    cb_reserve_back(out, n);
    tile_regs_acquire();
    mul_bcast_cols_init(src, cb_r);
    for (uint32_t t = 0; t < n; t++) {
        mul_tiles_bcast_cols(src, cb_r, base + t, 0, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < n; t++) {
        pack_tile(t, out, t);
    }
    tile_regs_release();
    cb_push_back(out, n);
    cb_pop_front(cb_r, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);       // key dim tiles (128 -> 4)
    constexpr uint32_t Vt = get_compile_time_arg_val(1);       // value dim tiles
    constexpr uint32_t NC = get_compile_time_arg_val(2);       // conv channel tiles of this head: 2 Kt + Vt
    constexpr uint32_t q_scale = get_compile_time_arg_val(3);  // fp32 bits of Dk^-0.5
    constexpr uint32_t l2_eps = get_compile_time_arg_val(4);   // fp32 bits (1e-6)
    constexpr uint32_t inv_dv = get_compile_time_arg_val(5);   // fp32 bits of 1 / Dv
    constexpr uint32_t rms_eps = get_compile_time_arg_val(6);  // fp32 bits (1e-6)
    constexpr uint32_t sp_beta = get_compile_time_arg_val(7);  // softplus beta / 1/beta / threshold bits
    constexpr uint32_t sp_thr = get_compile_time_arg_val(8);
    static_assert(Kt <= DST_TILES && Vt <= DST_TILES, "head dims up to 128");
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t ONE = 0x3f800000u;

    // ---------------- A. conv + silu (bf16 operands, fp32 out) ----------------
    cb_wait_front(cb_h, 3 * NC);
    cb_wait_front(cb_x, NC);
    cb_wait_front(cb_tap, 4 * NC);
    compute_kernel_hw_startup(cb_h, cb_tap, cb_c);
    cb_reserve_back(cb_c, NC);
    for (uint32_t i = 0; i < NC; i++) {
        tile_regs_acquire();
        mul_init(cb_h, cb_tap);
        mul_tiles(cb_h, cb_tap, i, i, 0);
        mul_tiles(cb_h, cb_tap, NC + i, NC + i, 1);
        mul_tiles(cb_h, cb_tap, 2 * NC + i, 2 * NC + i, 2);
        mul_init(cb_x, cb_tap);
        mul_tiles(cb_x, cb_tap, i, 3 * NC + i, 3);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(2, 3, 2);
        add_binary_tile(0, 2, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_c, i);
        tile_regs_release();
    }
    cb_push_back(cb_c, NC);
    cb_pop_front(cb_h, 3 * NC);
    cb_pop_front(cb_tap, 4 * NC);
    cb_pop_front(cb_x, NC);
    // sz = silu(z)
    cb_wait_front(cb_z, Vt);
    compute_kernel_hw_startup(cb_z, cb_sz);
    cb_reserve_back(cb_sz, Vt);
    tile_regs_acquire();
    copy_init(cb_z);
    for (uint32_t t = 0; t < Vt; t++) {
        copy_tile(cb_z, t, t);
    }
    silu_tile_init();
    for (uint32_t t = 0; t < Vt; t++) {
        silu_tile(t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < Vt; t++) {
        pack_tile(t, cb_sz, t);
    }
    tile_regs_release();
    cb_push_back(cb_sz, Vt);
    cb_pop_front(cb_z, Vt);

    // ---------------- B. l2norm q (x scale), k; v with padding rows zeroed ----------------
    cb_wait_front(cb_c, NC);
    cb_wait_front(cb_one, 1);
    cb_wait_front(cb_mcol, 1);
    sum_squares(cb_c, 0, Kt);
    row_rsqrt(ONE, l2_eps, q_scale);
    scale_rows(cb_c, 0, Kt, cb_qn);
    sum_squares(cb_c, Kt, Kt);
    row_rsqrt(ONE, l2_eps, ONE);
    scale_rows(cb_c, Kt, Kt, cb_kn);
    // v * [1 at (0,0)] column-broadcast: row 0 kept, rows 1..31 zero
    compute_kernel_hw_startup(cb_c, cb_mcol, cb_v);
    cb_reserve_back(cb_v, Vt);
    tile_regs_acquire();
    mul_bcast_cols_init(cb_c, cb_mcol);
    for (uint32_t t = 0; t < Vt; t++) {
        mul_tiles_bcast_cols(cb_c, cb_mcol, 2 * Kt + t, 0, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < Vt; t++) {
        pack_tile(t, cb_v, t);
    }
    tile_regs_release();
    cb_push_back(cb_v, Vt);
    cb_pop_front(cb_c, NC);

    // ---------------- C. gates ----------------
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_wait_front(cb_na, 1);
    cb_wait_front(cb_dt, 1);
    compute_kernel_hw_startup(cb_a, cb_decay);
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
    cb_pop_front(cb_na, 1);
    cb_pop_front(cb_dt, 1);

    // ---------------- D. recurrence (T = 1) ----------------
    compute_kernel_hw_startup(cb_S, cb_decay, cb_sd);
    cb_wait_front(cb_decay, 1);
    cb_wait_front(cb_S, kv);
    bcast_scalar_mul(cb_S, cb_decay, cb_sd, kv);
    cb_pop_front(cb_decay, 1);
    cb_pop_front(cb_S, kv);
    cb_wait_front(cb_sd, kv);
    cb_wait_front(cb_kn, Kt);
    mm(cb_kn, cb_sd, cb_vread, 1, Kt, Vt, false);
    cb_wait_front(cb_vread, Vt);
    cb_wait_front(cb_v, Vt);
    cb_wait_front(cb_beta, 1);
    delta_rule_residual(cb_v, cb_vread, cb_beta, cb_u, Vt);
    cb_pop_front(cb_v, Vt);
    cb_pop_front(cb_vread, Vt);
    cb_pop_front(cb_beta, 1);
    cb_wait_front(cb_u, Vt);
    transpose_block(cb_kn, cb_kcol, Kt);
    cb_pop_front(cb_kn, Kt);
    cb_wait_front(cb_kcol, Kt);
    rank1_update<Kt, Vt>(cb_sd, cb_kcol, cb_u, cb_snew);
    cb_pop_front(cb_kcol, Kt);
    cb_pop_front(cb_u, Vt);
    cb_pop_front(cb_sd, kv);
    cb_wait_front(cb_snew, kv);
    cb_wait_front(cb_qn, Kt);
    mm(cb_qn, cb_snew, cb_o, 1, Kt, Vt, false);
    cb_pop_front(cb_qn, Kt);
    cb_pop_front(cb_snew, kv);

    // ---------------- E. gated RMSNorm ----------------
    cb_wait_front(cb_o, Vt);
    sum_squares(cb_o, 0, Vt);
    row_rsqrt(inv_dv, rms_eps, ONE);
    cb_wait_front(cb_r, 1);
    cb_wait_front(cb_w, Vt);
    cb_wait_front(cb_sz, Vt);
    compute_kernel_hw_startup(cb_o, cb_r, cb_out);
    cb_reserve_back(cb_out, Vt);
    for (uint32_t t = 0; t < Vt; t++) {
        tile_regs_acquire();
        mul_bcast_cols_init(cb_o, cb_r);
        mul_tiles_bcast_cols(cb_o, cb_r, t, 0, 0);
        copy_init(cb_w);
        copy_tile(cb_w, t, 1);
        copy_init(cb_sz);
        copy_tile(cb_sz, t, 2);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, t);
        tile_regs_release();
    }
    cb_push_back(cb_out, Vt);
    cb_pop_front(cb_o, Vt);
    cb_pop_front(cb_r, 1);
    cb_pop_front(cb_w, Vt);
    cb_pop_front(cb_sz, Vt);
    cb_pop_front(cb_one, 1);
    cb_pop_front(cb_mcol, 1);
}
