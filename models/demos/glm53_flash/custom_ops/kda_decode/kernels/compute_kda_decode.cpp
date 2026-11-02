// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel of the fused KDA (Kimi Delta Attention) single-token decode block. One work item =
// (head h, 32-user tile row tb, user part). On the packed matmul outputs (row b = user b, columns of head h)
// the core computes for all 32 users at once:
//
//   conv   = SiLU(sum_j taps[j] * hist[j])  (hist[3] = the raw token)   per branch q, k, v  ([32, D] blocks)
//   q      = l2norm(conv_q) * K^-0.5 ;  k = l2norm(conv_k)              (FLA: x / sqrt(sum x^2 + 1e-6))
//   decay  = exp(lb * sigmoid(a_exp * (f + dt_bias)))                   per key channel  ([32, K])
//   beta   = sigmoid(b_logit)  (column) ;  wg = sigmoid(og) * w          ([32, V])
//   kd = k * decay ; qd = q * decay ; qk = rowsum(q * k) ; kcol = k^T ; decT = decay^T
//
// then per user r of the part, on its state S (no row lifting: a full-tile op's row r is user r's result):
//
//   vread = kd . S  (row r) ;  u = beta_r (v - vread) masked to row r ;  S' = diag(decay_r) S + k_r^T (x) u_r
//   o_r   = q_r S' = qd_r . S + (q_r . k_r) u_r   -> accumulated at row r of the part's output block
//
// S' goes to the writer in place; the part's output block (rows of its users, zero elsewhere) is summed on
// the slot's part-0 core (gather CB), which applies the gated RMSNorm (rmsnorm(o) * w * sigmoid(og)) and
// packs the bf16 TILE output block. Row sums use a matmul with an all-ones tile (every output column holds
// the row sum); per-row scales are SFPU column broadcasts; diag(decay_r) S is a column broadcast with the
// decay column (decT . E_r^T); masks are column broadcasts with E_r^T (e_r in column 0).
//
// Formats: history / raw / taps / selector tiles bf16, everything else fp32 (fp32 DST, half sync: 4 tiles).
// SFPU work (SiLU, sigmoid, exp, rsqrt, scalar / binary ops) runs only over the live rows of a tile row when
// it has <= 16 users: faces 0 and 1 (VectorMode::R) with ceil(rows / 2) SFPU iterations per face; the
// unprocessed rows keep finite values that the masks / selectors zero out.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

// per-phase profiler zones: only where the kernel profiler provides the macro (compiled out otherwise)
#if defined(PROFILE_KERNEL) && defined(DeviceZoneScopedN)
#define KDA_ZONE(name) DeviceZoneScopedN(name)
#else
#define KDA_ZONE(name)
#endif

namespace {

// CB plan (kept in sync with op.py and the dataflow kernels)
constexpr uint32_t cb_hist = 0;        // bf16 12*Kt: block (branch*4 + j)*Kt + t; j = 0..2 history taps, 3 = raw
constexpr uint32_t cb_taps = 1;        // bf16 3*Kt: branch b tile t at b*Kt + t, rows 0..3 = taps
constexpr uint32_t cb_gf = 2;          // fp32 Kt: forget-gate pre-activation block [32, K]
constexpr uint32_t cb_gog = 3;         // fp32 Vt: output-gate pre-activation block [32, V]
constexpr uint32_t cb_gb = 4;          // fp32 1: beta logits tile (column c = h % 32)
constexpr uint32_t cb_cc = 5;          // fp32 Kt: channel constants (row 0 dt_bias, row 1 exp(A_log))
constexpr uint32_t cb_onw = 6;         // fp32 Vt: o_norm weight (row 0), once per core
constexpr uint32_t cb_sel = 7;         // bf16 2: all-ones tile, identity tile; once per core
constexpr uint32_t cb_selT = 8;        // bf16 1 per user: E_r^T (1 at (r, 0))
constexpr uint32_t cb_S = 9;           // fp32 kv per user: recurrent state
constexpr uint32_t cb_out = 10;        // bf16 Vt: the head's normed output tiles (part-0 core)
constexpr uint32_t cb_state_out = 11;  // fp32 kv per user: updated state
constexpr uint32_t cb_qn = 12;         // fp32 Kt: q after conv + l2norm * K^-0.5
constexpr uint32_t cb_kn = 13;         // fp32 Kt: k after conv + l2norm
constexpr uint32_t cb_vc = 14;         // fp32 Vt: v after conv
constexpr uint32_t cb_dec = 15;        // fp32 Kt: decay block [32, K]
constexpr uint32_t cb_bcol = 16;       // fp32 1: sigmoid(beta logits) column (column 0)
constexpr uint32_t cb_wg = 17;         // fp32 Vt: sigmoid(og) * w
constexpr uint32_t cb_tmp = 18;        // fp32 2*Kt: scratch (two 4-tile slots)
constexpr uint32_t cb_col = 19;        // fp32 2: per-row scalars as column tiles (inv norms, then q.k)
constexpr uint32_t cb_kd = 20;         // fp32 Kt: k * decay
constexpr uint32_t cb_qd = 21;         // fp32 Kt: q * decay
constexpr uint32_t cb_kcol = 22;       // fp32 Kt: k^T (user b's key in column b)
constexpr uint32_t cb_decT = 23;       // fp32 Kt: decay^T
constexpr uint32_t cb_gather = 24;     // fp32 (parts-1)*Vt: the other parts' output blocks (part-0 core)
constexpr uint32_t cb_dcol = 25;       // fp32 Kt per user: decay_r as a column (column 0)
constexpr uint32_t cb_pq = 26;         // fp32 4 per user: beta_r e_r | diag(e_r) | (q.k)_r e_r | (q.k)_r diag(e_r)
constexpr uint32_t cb_vread = 27;      // fp32 Vt: kd . S
constexpr uint32_t cb_oq = 28;         // fp32 Vt: qd . S
constexpr uint32_t cb_u = 29;          // fp32 Vt: beta_r (v - vread) at row r
constexpr uint32_t cb_oacc = 30;       // fp32 2*Vt: output block accumulator (two slots)
constexpr uint32_t cb_ofin = 31;       // fp32 Vt: the part's final output block

constexpr uint32_t DST_TILES = 4;  // fp32 DST tiles per acquire (half-sync)
constexpr uint32_t ONES = 0;       // selector CB tile indices
constexpr uint32_t IDENT = 1;

inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---------------------------------------------------------------- SFPU over the live rows only
// IT_H = SFPU iterations per face (2 rows each) when a tile row has <= 16 live users; full tiles otherwise.
template <uint32_t IT_H>
struct Sfpu {
    static void silu(uint32_t d, bool half) {
        if (half) {
            MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, true, calculate_silu, (true, IT_H), d, VectorMode::R));
        } else {
            silu_tile(d);
        }
    }
    static void sigmoid(uint32_t d, bool half) {
        if (half) {
            MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, true, calculate_sigmoid, (false, true, IT_H), d, VectorMode::R));
        } else {
            sigmoid_tile(d);
        }
    }
    static void exp(uint32_t d, bool half) {
        if (half) {
            exp_tile<false, false, InputClamping::ClampToNegative, IT_H, true>(d, VectorMode::R);
        } else {
            exp_tile(d);
        }
    }
    static void rsqrt(uint32_t d, bool half) {
        if (half) {
            MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, true, calculate_rsqrt, (false, IT_H, true, false), d, VectorMode::R));
        } else {
            rsqrt_tile(d);
        }
    }
    static void mul_scalar(uint32_t d, uint32_t bits, bool half) {
        if (half) {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                true,
                calculate_binop_with_scalar,
                (APPROX, MUL_UNARY, IT_H, true),
                d,
                VectorMode::R,
                bits));
        } else {
            mul_unary_tile(d, bits);
        }
    }
    static void add_scalar(uint32_t d, uint32_t bits, bool half) {
        if (half) {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                true,
                calculate_binop_with_scalar,
                (APPROX, ADD_UNARY, IT_H, true),
                d,
                VectorMode::R,
                bits));
        } else {
            add_unary_tile(d, bits);
        }
    }
    static void add(uint32_t a, uint32_t b, uint32_t o, bool half) {
        if (half) {
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                true,
                calculate_sfpu_binary,
                (APPROX, ckernel::BinaryOp::ADD, IT_H, true),
                a,
                b,
                o,
                VectorMode::R)));
        } else {
            add_binary_tile(a, b, o);
        }
    }
    static void mul(uint32_t a, uint32_t b, uint32_t o, bool half) {
        if (half) {
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                true,
                calculate_sfpu_binary,
                (APPROX, ckernel::BinaryOp::MUL, IT_H, true),
                a,
                b,
                o,
                VectorMode::R)));
        } else {
            mul_binary_tile(a, b, o);
        }
    }
};

// out[n] = A[a_off .. a_off+Kt) (one tile row) @ B[Kt, Nt], n < Nt (Nt <= DST_TILES)
template <uint32_t Kt, uint32_t Nt>
void mm_row(uint32_t a, uint32_t a_off, uint32_t b, uint32_t o) {
    static_assert(Nt <= DST_TILES);
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Nt);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b, 0);
    tile_regs_acquire();
    for (uint32_t ni = 0; ni < Nt; ni++) {
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(a, b, a_off + ki, ki * Nt + ni, ni);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t ni = 0; ni < Nt; ni++) {
        pack_tile<true>(ni, o, ni);
    }
    tile_regs_release();
    cb_o.push_back(Nt);
}

// out[i] = transpose(in[in_off + i]), n tiles (n <= DST_TILES)
void transpose_block(uint32_t in, uint32_t in_off, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    reconfig_data_format_srca(in);
    transpose_init(in);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; i++) {
        transpose_tile(in, in_off + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < n; i++) {
        pack_tile<true>(i, o, i);
    }
    tile_regs_release();
    cb_o.push_back(n);
}

// col = f(row sums of X[x_off .. x_off+n)) as a tile whose every column holds the per-row value:
//   sums via matmul with the all-ones tile, then  sums * mul_pre + eps -> rsqrt -> * mul_post
template <uint32_t IT_H>
void row_inv_norm(
    uint32_t x, uint32_t x_off, uint32_t n, uint32_t mul_pre, uint32_t eps, uint32_t mul_post, bool half) {
    using SF = Sfpu<IT_H>;
    CircularBuffer cb_o(cb_col);
    cb_o.reserve_back(1);
    reconfig_data_format<SrcOrder::Reverse>(x, cb_sel);
    matmul_init(x, cb_sel, 0);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; i++) {
        matmul_tiles(x, cb_sel, x_off + i, ONES, 0);
    }
    binop_with_scalar_tile_init();
    if (mul_pre != 0x3F800000u) {
        SF::mul_scalar(0, mul_pre, half);
    }
    SF::add_scalar(0, eps, half);
    rsqrt_tile_init();
    SF::rsqrt(0, half);
    if (mul_post != 0x3F800000u) {
        binop_with_scalar_tile_init();
        SF::mul_scalar(0, mul_post, half);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_col);
    tile_regs_release();
    cb_o.push_back(1);
}

// o[i] = A[a_off + i] * B[b_off (+ i)] elementwise, n tiles (n <= DST_TILES)
void mul_blocks(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, bool b_same, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    reconfig_data_format(a, b);
    mul_init(a, b);
    tile_regs_acquire();
    for (uint32_t j = 0; j < n; j++) {
        mul_tiles(a, b, a_off + j, b_same ? b_off : b_off + j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < n; j++) {
        pack_tile<true>(j, o, j);
    }
    tile_regs_release();
    cb_o.push_back(n);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t parts = get_compile_time_arg_val(4);         // user split of a tile row over cores
    constexpr uint32_t lb_bits = get_compile_time_arg_val(5);       // gate lower bound (-5)
    constexpr uint32_t l2eps_bits = get_compile_time_arg_val(6);    // 1e-6
    constexpr uint32_t normeps_bits = get_compile_time_arg_val(7);  // o_norm eps
    constexpr uint32_t qscale_bits = get_compile_time_arg_val(8);   // K^-0.5
    constexpr uint32_t invv_bits = get_compile_time_arg_val(9);     // 1/V
    constexpr uint32_t IT_H = get_compile_time_arg_val(10);         // SFPU iterations per face for <= 16-user rows
    using SF = Sfpu<IT_H>;
    constexpr uint32_t ONE_BITS = 0x3F800000u;
    static_assert(Kt == DST_TILES && Vt == DST_TILES, "the kernel is written for K = V = 128 (4-tile blocks)");
    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);

    constexpr uint32_t kv = Kt * Vt;

    // fp32 everywhere by default; the bf16 CBs are reconfigured for explicitly where they are read / written
    compute_kernel_hw_startup(cb_tmp, cb_tmp, cb_tmp);

    CircularBuffer hist(cb_hist), taps(cb_taps), gf(cb_gf), gog(cb_gog), gb(cb_gb), cc(cb_cc), onw(cb_onw);
    CircularBuffer sel(cb_sel), selT(cb_selT), S(cb_S), out(cb_out), st_out(cb_state_out);
    CircularBuffer qn(cb_qn), kn(cb_kn), vc(cb_vc), dec(cb_dec), bcol(cb_bcol), wg(cb_wg), tmp(cb_tmp);
    CircularBuffer col(cb_col), kd(cb_kd), qd(cb_qd), kcol(cb_kcol), decT(cb_decT), gather(cb_gather);
    CircularBuffer dcol(cb_dcol), pq(cb_pq), vread(cb_vread), oq(cb_oq), u(cb_u), oacc(cb_oacc), ofin(cb_ofin);

    onw.wait_front(Vt);
    sel.wait_front(2);

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t n_row = min_u(32u, B - tb * 32u);  // live users of this tile row
        const uint32_t up = (n_row + parts - 1) / parts;  // users per part
        const uint32_t n_active = (n_row + up - 1) / up;
        const uint32_t u0 = part * up;
        const uint32_t u1 = min_u(u0 + up, n_row);
        if (u0 >= u1) {
            continue;  // nothing for this part (the reader/writer skip it the same way)
        }
        const bool half = n_row <= 16u;  // SFPU over faces 0 / 1 only

        taps.wait_front(3 * Kt);
        hist.wait_front(12 * Kt);
        // ------------------------------------------------------------ 1. conv + SiLU per branch (+ l2 norms)
        {
            KDA_ZONE("kda_conv");
            for (uint32_t br = 0; br < 3; br++) {
                const bool l2 = br < 2;
                const uint32_t out_cb = l2 ? cb_tmp : cb_vc;
                CircularBuffer cb_o(out_cb);
                cb_o.reserve_back(l2 ? 2 * Kt : Kt);
                reconfig_data_format(cb_hist, cb_taps);
                mul_bcast_rows_init(cb_hist, cb_taps);
                for (uint32_t i = 0; i < Kt; i++) {
                    tile_regs_acquire();
                    // conv[i] = sum_j hist[j][i] * taps[j]  (row broadcasts of the tap rows), then SiLU
                    for (uint32_t j = 0; j < 4; j++) {
                        mul_tiles_bcast_rows(cb_hist, cb_taps, (br * 4 + j) * Kt + i, br * Kt + i, j, j);
                    }
                    add_binary_tile_init();
                    SF::add(0, 1, 0, half);
                    SF::add(2, 3, 2, half);
                    SF::add(0, 2, 0, half);
                    silu_tile_init();
                    SF::silu(0, half);
                    if (l2) {  // squares for the row sums
                        mul_binary_tile_init();
                        SF::mul(0, 0, 1, half);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, out_cb, i);
                    if (l2) {
                        pack_tile<true>(1, cb_tmp, Kt + i);
                    }
                    tile_regs_release();
                }
                cb_o.push_back(l2 ? 2 * Kt : Kt);
                if (l2) {
                    // l2norm over the K columns of every user row: inv = rsqrt(sum x^2 + eps) [* K^-0.5 for q]
                    tmp.wait_front(2 * Kt);
                    row_inv_norm<IT_H>(cb_tmp, Kt, Kt, ONE_BITS, l2eps_bits, br == 0 ? qscale_bits : ONE_BITS, half);
                    col.wait_front(1);
                    mul_blocks(cb_tmp, 0, cb_col, 0, true, br == 0 ? cb_qn : cb_kn, Kt);
                    col.pop_front(1);
                    tmp.pop_front(2 * Kt);
                }
            }
        }
        taps.pop_front(3 * Kt);
        hist.pop_front(12 * Kt);
        qn.wait_front(Kt);
        kn.wait_front(Kt);
        vc.wait_front(Vt);

        // ------------------------------------------------------------ 2. gates
        {
            KDA_ZONE("kda_gates");
            gf.wait_front(Kt);
            cc.wait_front(Kt);
            // f + dt_bias (row 0 of cc), then * a_exp (row 1 of cc), sigmoid, * lb, exp -> decay block
            tmp.reserve_back(Kt);
            reconfig_data_format(cb_gf, cb_cc);
            add_bcast_rows_init(cb_gf, cb_cc);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Kt; i++) {
                add_tiles_bcast_rows(cb_gf, cb_cc, i, i, i, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Kt; i++) {
                pack_tile<true>(i, cb_tmp, i);
            }
            tile_regs_release();
            tmp.push_back(Kt);
            tmp.wait_front(Kt);
            dec.reserve_back(Kt);
            reconfig_data_format(cb_tmp, cb_cc);
            mul_bcast_rows_init(cb_tmp, cb_cc);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Kt; i++) {
                mul_tiles_bcast_rows(cb_tmp, cb_cc, i, i, i, 1);
            }
            sigmoid_tile_init();
            for (uint32_t i = 0; i < Kt; i++) {
                SF::sigmoid(i, half);
            }
            binop_with_scalar_tile_init();
            for (uint32_t i = 0; i < Kt; i++) {
                SF::mul_scalar(i, lb_bits, half);
            }
            exp_tile_init();
            for (uint32_t i = 0; i < Kt; i++) {
                SF::exp(i, half);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Kt; i++) {
                pack_tile<true>(i, cb_dec, i);
            }
            tile_regs_release();
            dec.push_back(Kt);
            tmp.pop_front(Kt);
            gf.pop_front(Kt);
            cc.pop_front(Kt);
            dec.wait_front(Kt);
            // beta column: gb . E_c^T puts column c of the logits tile into column 0; sigmoid
            gb.wait_front(1);
            selT.wait_front(1);  // E_c^T (the reader queues it ahead of the users' tiles)
            bcol.reserve_back(1);
            reconfig_data_format<SrcOrder::Reverse>(cb_gb, cb_selT);
            matmul_init(cb_gb, cb_selT, 0);
            tile_regs_acquire();
            matmul_tiles(cb_gb, cb_selT, 0, 0, 0);
            sigmoid_tile_init();
            SF::sigmoid(0, half);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcol);
            tile_regs_release();
            selT.pop_front(1);
            bcol.push_back(1);
            gb.pop_front(1);
            bcol.wait_front(1);
            // output gate * w: wg = sigmoid(og) * w (row broadcast of the weight row)
            gog.wait_front(Vt);
            tmp.reserve_back(Vt);
            reconfig_data_format_srca(cb_gog);
            copy_init(cb_gog);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Vt; i++) {
                copy_tile(cb_gog, i, i);
            }
            sigmoid_tile_init();
            for (uint32_t i = 0; i < Vt; i++) {
                SF::sigmoid(i, half);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Vt; i++) {
                pack_tile<true>(i, cb_tmp, i);
            }
            tile_regs_release();
            tmp.push_back(Vt);
            tmp.wait_front(Vt);
            wg.reserve_back(Vt);
            reconfig_data_format(cb_tmp, cb_onw);
            mul_bcast_rows_init(cb_tmp, cb_onw);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Vt; i++) {
                mul_tiles_bcast_rows(cb_tmp, cb_onw, i, i, i, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Vt; i++) {
                pack_tile<true>(i, cb_wg, i);
            }
            tile_regs_release();
            wg.push_back(Vt);
            tmp.pop_front(Vt);
            gog.pop_front(Vt);
        }
        // ------------------------------------------------------------ 3. per-item derived blocks
        {
            KDA_ZONE("kda_derived");
            mul_blocks(cb_kn, 0, cb_dec, 0, false, cb_kd, Kt);    // kd = k * decay
            mul_blocks(cb_qn, 0, cb_dec, 0, false, cb_qd, Kt);    // qd = q * decay
            mul_blocks(cb_qn, 0, cb_kn, 0, false, cb_vread, Kt);  // q * k (vread as scratch) -> row sums = q . k
            vread.wait_front(Kt);
            {
                col.reserve_back(1);
                reconfig_data_format<SrcOrder::Reverse>(cb_vread, cb_sel);
                matmul_init(cb_vread, cb_sel, 0);
                tile_regs_acquire();
                for (uint32_t i = 0; i < Kt; i++) {
                    matmul_tiles(cb_vread, cb_sel, i, ONES, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_col);
                tile_regs_release();
                col.push_back(1);
            }
            vread.pop_front(Kt);
            transpose_block(cb_kn, 0, cb_kcol, Kt);   // kcol: user b's key in column b
            transpose_block(cb_dec, 0, cb_decT, Kt);  // decT: user b's decay in column b
            kd.wait_front(Kt);
            qd.wait_front(Kt);
            col.wait_front(1);
            kcol.wait_front(Kt);
            decT.wait_front(Kt);
        }

        // ------------------------------------------------------------ 4. recurrence per user
        for (uint32_t r = u0; r < u1; r++) {
            KDA_ZONE("kda_user");
            const bool last = (r + 1 == u1);
            selT.wait_front(1);
            S.wait_front(kv);
            // dcol = decT . E_r^T : decay_r down column 0 of every tile
            {
                dcol.reserve_back(Kt);
                reconfig_data_format<SrcOrder::Reverse>(cb_decT, cb_selT);
                matmul_init(cb_decT, cb_selT, 0);
                tile_regs_acquire();
                for (uint32_t i = 0; i < Kt; i++) {
                    matmul_tiles(cb_decT, cb_selT, i, 0, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < Kt; i++) {
                    pack_tile<true>(i, cb_dcol, i);
                }
                tile_regs_release();
                dcol.push_back(Kt);
            }
            // pq: bm = beta_r e_r (col 0) | P = diag(e_r) | qm = (q.k)_r e_r | Q = (q.k)_r diag(e_r)
            {
                pq.reserve_back(3);
                tile_regs_acquire();
                reconfig_data_format(cb_bcol, cb_selT);
                mul_bcast_cols_init(cb_bcol, cb_selT);
                mul_tiles_bcast_cols(cb_bcol, cb_selT, 0, 0, 0);
                reconfig_data_format(cb_sel, cb_selT);
                mul_bcast_cols_init(cb_sel, cb_selT);
                mul_tiles_bcast_cols(cb_sel, cb_selT, IDENT, 0, 1);
                reconfig_data_format(cb_selT, cb_col);
                mul_bcast_cols_init(cb_selT, cb_col);
                mul_tiles_bcast_cols(cb_selT, cb_col, 0, 0, 2);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < 3; i++) {
                    pack_tile<true>(i, cb_pq, i);
                }
                tile_regs_release();
                pq.push_back(3);
                pq.wait_front(3);
                pq.reserve_back(1);
                reconfig_data_format(cb_sel, cb_pq);
                mul_bcast_cols_init(cb_sel, cb_pq);
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_sel, cb_pq, IDENT, 2, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_pq);
                tile_regs_release();
                pq.push_back(1);
                pq.wait_front(4);
            }
            dcol.wait_front(Kt);
            // vread = kd . S ; oq = qd . S   (row r of each is user r's)
            mm_row<Kt, Vt>(cb_kd, 0, cb_S, cb_vread);
            mm_row<Kt, Vt>(cb_qd, 0, cb_S, cb_oq);
            vread.wait_front(Vt);
            oq.wait_front(Vt);
            // u = beta_r e_r (col bcast) * (v - vread): row r live, other rows zero
            {
                u.reserve_back(Vt);
                for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES - 1) {
                    const uint32_t nn = min_u(Vt - n0, DST_TILES - 1);
                    tile_regs_acquire();
                    reconfig_data_format(cb_vc, cb_vread);
                    sub_init(cb_vc, cb_vread);
                    for (uint32_t j = 0; j < nn; j++) {
                        sub_tiles(cb_vc, cb_vread, n0 + j, n0 + j, j);
                    }
                    reconfig_data_format_srca(cb_pq);
                    copy_init(cb_pq);
                    copy_tile(cb_pq, 0, nn);
                    sfpu_mul_bcast_col_init();
                    for (uint32_t j = 0; j < nn; j++) {
                        sfpu_mul_bcast_col(j, nn);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < nn; j++) {
                        pack_tile<true>(j, cb_u, n0 + j);
                    }
                    tile_regs_release();
                }
                u.push_back(Vt);
            }
            vread.pop_front(Vt);
            u.wait_front(Vt);
            // S' = diag(decay_r) S + k_r^T (x) u_r  -> the state output (one pack per tile)
            {
                st_out.reserve_back(kv);
                for (uint32_t mi = 0; mi < Kt; mi++) {
                    tile_regs_acquire();
                    reconfig_data_format(cb_S, cb_dcol);
                    mul_bcast_cols_init(cb_S, cb_dcol);
                    for (uint32_t j = 0; j < Vt; j++) {
                        mul_tiles_bcast_cols(cb_S, cb_dcol, mi * Vt + j, mi, j);
                    }
                    reconfig_data_format<SrcOrder::Reverse>(cb_kcol, cb_u);
                    matmul_init(cb_kcol, cb_u, 0);
                    for (uint32_t j = 0; j < Vt; j++) {
                        matmul_tiles(cb_kcol, cb_u, mi, j, j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < Vt; j++) {
                        pack_tile<true>(j, cb_state_out, mi * Vt + j);
                    }
                    tile_regs_release();
                }
                st_out.push_back(kv);
            }
            S.pop_front(kv);
            dcol.pop_front(Kt);
            // output block: oacc += diag(e_r) oq + (q.k)_r diag(e_r) u   (row r of oq / u)
            {
                const uint32_t o_cb = last ? cb_ofin : cb_oacc;
                CircularBuffer cb_o(o_cb);
                cb_o.reserve_back(Vt);
                tile_regs_acquire();
                if (r > u0) {
                    reconfig_data_format_srca(cb_oacc);
                    copy_init(cb_oacc);
                    for (uint32_t n = 0; n < Vt; n++) {
                        copy_tile(cb_oacc, n, n);
                    }
                }
                reconfig_data_format<SrcOrder::Reverse>(cb_pq, cb_oq);
                matmul_init(cb_pq, cb_oq, 0);
                for (uint32_t n = 0; n < Vt; n++) {
                    matmul_tiles(cb_pq, cb_oq, 1, n, n);
                    matmul_tiles(cb_pq, cb_u, 3, n, n);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t n = 0; n < Vt; n++) {
                    pack_tile<true>(n, o_cb, n);
                }
                tile_regs_release();
                cb_o.push_back(Vt);
                if (r > u0) {
                    oacc.pop_front(Vt);
                }
            }
            oq.pop_front(Vt);
            u.pop_front(Vt);
            pq.pop_front(4);
            selT.pop_front(1);
        }
        qn.pop_front(Kt);
        kn.pop_front(Kt);
        vc.pop_front(Vt);
        dec.pop_front(Kt);
        bcol.pop_front(1);
        kd.pop_front(Kt);
        qd.pop_front(Kt);
        kcol.pop_front(Kt);
        decT.pop_front(Kt);
        col.pop_front(1);

        // ------------------------------------------------------------ 5. part-0 core: sum the parts, gated RMSNorm
        if (part == 0) {
            KDA_ZONE("kda_out_norm");
            ofin.wait_front(Vt);
            const uint32_t n_other = (parts > 1) ? (n_active - 1) : 0;
            if (n_other > 0) {
                gather.wait_front(n_other * Vt);
            }
            // o = sum of the parts' blocks -> tmp[0..Vt), o^2 -> tmp[Vt..2Vt)
            tmp.reserve_back(2 * Vt);
            for (uint32_t n0 = 0; n0 < Vt; n0 += 2) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_ofin);
                copy_init(cb_ofin);
                copy_tile(cb_ofin, n0, 0);
                copy_tile(cb_ofin, n0 + 1, 1);
                if (n_other > 0) {
                    reconfig_data_format_srca(cb_gather);
                    copy_init(cb_gather);
                    add_binary_tile_init();
                    for (uint32_t p = 0; p < n_other; p++) {
                        copy_tile(cb_gather, p * Vt + n0, 2);
                        copy_tile(cb_gather, p * Vt + n0 + 1, 3);
                        SF::add(0, 2, 0, half);
                        SF::add(1, 3, 1, half);
                    }
                }
                mul_binary_tile_init();
                SF::mul(0, 0, 2, half);
                SF::mul(1, 1, 3, half);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_tmp, n0);
                pack_tile<true>(1, cb_tmp, n0 + 1);
                pack_tile<true>(2, cb_tmp, Vt + n0);
                pack_tile<true>(3, cb_tmp, Vt + n0 + 1);
                tile_regs_release();
            }
            tmp.push_back(2 * Vt);
            tmp.wait_front(2 * Vt);
            if (n_other > 0) {
                gather.pop_front(n_other * Vt);
            }
            ofin.pop_front(Vt);
            row_inv_norm<IT_H>(cb_tmp, Vt, Vt, invv_bits, normeps_bits, ONE_BITS, half);  // rsqrt(mean(o^2) + eps)
            col.wait_front(1);
            wg.wait_front(Vt);
            // out = (o * wg) * inv (column broadcast), packed as bf16 tiles
            out.reserve_back(Vt);
            pack_reconfig_data_format(cb_out);
            for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES - 1) {
                const uint32_t nn = min_u(Vt - n0, DST_TILES - 1);
                tile_regs_acquire();
                reconfig_data_format(cb_tmp, cb_wg);
                mul_init(cb_tmp, cb_wg);
                for (uint32_t j = 0; j < nn; j++) {
                    mul_tiles(cb_tmp, cb_wg, n0 + j, n0 + j, j);
                }
                reconfig_data_format_srca(cb_col);
                copy_init(cb_col);
                copy_tile(cb_col, 0, nn);
                sfpu_mul_bcast_col_init();
                for (uint32_t j = 0; j < nn; j++) {
                    sfpu_mul_bcast_col(j, nn);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < nn; j++) {
                    pack_tile<true>(j, cb_out, n0 + j);
                }
                tile_regs_release();
            }
            pack_reconfig_data_format(cb_tmp);
            out.push_back(Vt);
            col.pop_front(1);
            tmp.pop_front(2 * Vt);
        }
        wg.pop_front(Vt);
    }
}
