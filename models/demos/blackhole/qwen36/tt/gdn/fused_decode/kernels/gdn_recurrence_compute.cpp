// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode recurrence compute: one (batch b, value-head vh) per core, single token.
// Matches models/experimental/gated_attention_gated_deltanet recurrent_gated_delta_rule_decode_ttnn
// (high_precision) plus the qwen36 head-prep and gated out-norm:
//
//   beta  = sigmoid(ab[b, NV+vh])
//   decay = exp(neg_exp_A[vh] * softplus(ab[b, vh] + dt_bias[vh]))
//   qn = l2norm(q_b) * Dk^-0.5 ; kn = l2norm(k_b)            (l2norm: x * rsqrt(sum(x^2)+eps))
//   hd = h * decay
//   delta = (v_b - kn_b @ hd) * beta
//   outer = kn_b^T (x) delta                                  (row b only, via bcast_row_idx)
//   h_new = hd + outer                                        (written back in place by the writer)
//   o = qn @ h_new = qn @ hd + qn @ outer                     (both terms accumulate in dest)
//   out_b = rmsnorm(o_b) * norm_w * silu(z_b)
//
// Tiles hold all batch rows; every step is row-wise (reductions produce per-row column
// scalars, scalar factors are materialized as full-broadcast tiles), so each row stays
// self-consistent and only row b is ever selected — by bcast_row_idx here and by the
// writer's partial-row output write.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_qin = 0, cb_kin = 1, cb_vin = 2, cb_zin = 3;
constexpr uint32_t cb_ab = 4, cb_dtb = 5, cb_nega = 6, cb_w = 7, cb_ones = 8;
constexpr uint32_t cb_h = 9, cb_beta_full = 10, cb_decay_full = 11, cb_scr = 12;
constexpr uint32_t cb_decay_s = 13, cb_beta_s = 14, cb_sq = 15, cb_colscale = 16;
constexpr uint32_t cb_qn = 17, cb_kn = 18, cb_hd = 19;
// cb_vread is reused as silu(z), cb_delta as the normed o, cb_dm as normed*weight.
constexpr uint32_t cb_vread = 20, cb_delta = 21, cb_dm = 22;
constexpr uint32_t cb_kcb = 23, cb_outer = 24, cb_hnew = 25, cb_o = 26, cb_out = 27;

namespace {

inline void WAIT(uint32_t cb, uint32_t n) { cb_wait_front(cb, n); }
inline void POP(uint32_t cb, uint32_t n) { cb_pop_front(cb, n); }

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    if (op == 0) {
        add_init(a, b);
    } else if (op == 1) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        if (op == 0) {
            add_tiles(a, b, i, i, 0);
        } else if (op == 1) {
            sub_tiles(a, b, i, i, 0);
        } else {
            mul_tiles(a, b, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = A * A elementwise (per-tile square), n tiles.
void square(uint32_t a, uint32_t o, uint32_t n) { ew(a, a, o, n, 2); }

// out = A * scalar, n tiles; scalar is element [0,0] of the single `scal` tile.
void bscalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = A * col[.,0] broadcast across columns, n tiles (per-row scale).
void bcols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);
    mul_bcast_cols_init(a, col);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(a, col, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = A * row `row_idx` of B tile `btile`, broadcast down rows. One output tile.
void brow_mul_one(uint32_t a, uint32_t b, uint32_t atile, uint32_t btile, uint32_t o, uint32_t otile, uint32_t row_idx) {
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    mul_bcast_rows_init(a, b);
    tile_regs_acquire();
    mul_tiles_bcast_rows(a, b, atile, btile, 0, row_idx);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, otile);
    tile_regs_release();
}

// Row-sum of `in` (n tiles wide) via ones-matmul, then in-dest SFPU post-ops, one output tile:
//   dest = sum_k in[k] @ ones ; [+ mul pre_mul] ; + add_bits ; rsqrt ; [ * post_mul ]
// pre_mul/post_mul of 0 skip that step (fp32 bit patterns are never 0 for real factors).
void rowsum_rsqrt(uint32_t in, uint32_t o, uint32_t n, uint32_t pre_mul, uint32_t add_bits, uint32_t post_mul) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_ones, in);  // matmul(in, ones): in->srcB, ones->srcA
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t k = 0; k < n; k++) {
        matmul_tiles(in, cb_ones, k, 0, 0);
    }
    binop_with_scalar_tile_init();
    if (pre_mul != 0) {
        mul_unary_tile(0, pre_mul);
    }
    add_unary_tile(0, add_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if (post_mul != 0) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, post_mul);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// out[n] = accumulate_m rowcb[m] @ mat[m*n_tiles + n] (+ optionally a second mat).
void mm_row_state(uint32_t rowcb, uint32_t mat, uint32_t mat2, uint32_t o, uint32_t kt, uint32_t nt) {
    cb_reserve_back(o, nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(mat, rowcb);  // matmul(rowcb, mat): rowcb->srcB, mat->srcA
    matmul_init(rowcb, mat, 0);
    for (uint32_t n = 0; n < nt; n++) {
        tile_regs_acquire();
        for (uint32_t m = 0; m < kt; m++) {
            matmul_tiles(rowcb, mat, m, m * nt + n, 0);
        }
        if (mat2 != mat) {
            for (uint32_t m = 0; m < kt; m++) {
                matmul_tiles(rowcb, mat2, m, m * nt + n, 0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, n);
        tile_regs_release();
    }
    cb_push_back(o, nt);
}

// Materialize the [b, col] element of `full` (via cb_scr) as a full-broadcast tile in `o`:
//   scr0 = ones *bcast_row(b) full   -> every row = full[b, :]
//   scr1 = transpose(scr0)           -> [r, c] = full[b, r]
//   o    = ones *bcast_row(col) scr1 -> every element = full[b, col]
void extract_scalar(uint32_t full, uint32_t o, uint32_t b, uint32_t col) {
    cb_reserve_back(cb_scr, 1);
    brow_mul_one(cb_ones, full, 0, 0, cb_scr, 0, b);
    cb_push_back(cb_scr, 1);

    WAIT(cb_scr, 1);
    cb_reserve_back(cb_scr, 1);
    pack_reconfig_data_format(cb_scr);
    reconfig_data_format_srca(cb_scr);
    transpose_init(cb_scr);
    tile_regs_acquire();
    transpose_tile(cb_scr, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_scr, 0);
    tile_regs_release();
    cb_push_back(cb_scr, 1);
    POP(cb_scr, 1);

    WAIT(cb_scr, 1);
    cb_reserve_back(o, 1);
    brow_mul_one(cb_ones, cb_scr, 0, 0, o, 0, col);
    cb_push_back(o, 1);
    POP(cb_scr, 1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t NV = get_named_compile_time_arg_val("nv");
    constexpr uint32_t DKT = get_named_compile_time_arg_val("dkt");
    constexpr uint32_t DVT = get_named_compile_time_arg_val("dvt");
    constexpr uint32_t EPS = get_named_compile_time_arg_val("eps_bits");
    constexpr uint32_t SCALE = get_named_compile_time_arg_val("scale_bits");
    constexpr uint32_t SP_BETA = get_named_compile_time_arg_val("sp_beta_bits");
    constexpr uint32_t SP_BETA_RECIP = get_named_compile_time_arg_val("sp_beta_recip_bits");
    constexpr uint32_t SP_THR = get_named_compile_time_arg_val("sp_thr_bits");
    constexpr uint32_t INV_DV = get_named_compile_time_arg_val("inv_dv_bits");
    constexpr uint32_t NORM_EPS = get_named_compile_time_arg_val("norm_eps_bits");

    const uint32_t b = get_arg_val<uint32_t>(0);
    const uint32_t vh = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_qin, cb_h, cb_out);

    // ---- per-(b,vh) gate scalars from the ab tile -------------------------------
    WAIT(cb_ab, 1);
    WAIT(cb_ones, 1);

    // beta_full = sigmoid(ab); beta lives at column NV + vh of row b.
    cb_reserve_back(cb_beta_full, 1);
    pack_reconfig_data_format(cb_beta_full);
    reconfig_data_format_srca(cb_ab);
    copy_tile_init(cb_ab);
    sigmoid_tile_init();
    tile_regs_acquire();
    copy_tile(cb_ab, 0, 0);
    sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_beta_full, 0);
    tile_regs_release();
    cb_push_back(cb_beta_full, 1);

    // decay_full = exp(neg_exp_A * softplus(ab + dt_bias)); a lives at column vh of row b.
    WAIT(cb_dtb, 1);
    cb_reserve_back(cb_scr, 1);
    pack_reconfig_data_format(cb_scr);
    reconfig_data_format(cb_ab, cb_dtb);
    add_bcast_rows_init(cb_ab, cb_dtb);
    softplus_tile_init();
    tile_regs_acquire();
    add_tiles_bcast_rows(cb_ab, cb_dtb, 0, 0, 0, 0);
    softplus_tile(0, SP_BETA, SP_BETA_RECIP, SP_THR);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_scr, 0);
    tile_regs_release();
    cb_push_back(cb_scr, 1);

    WAIT(cb_nega, 1);
    WAIT(cb_scr, 1);
    cb_reserve_back(cb_decay_full, 1);
    pack_reconfig_data_format(cb_decay_full);
    reconfig_data_format(cb_scr, cb_nega);
    mul_bcast_rows_init(cb_scr, cb_nega);
    exp_tile_init();
    tile_regs_acquire();
    mul_tiles_bcast_rows(cb_scr, cb_nega, 0, 0, 0, 0);
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_decay_full, 0);
    tile_regs_release();
    cb_push_back(cb_decay_full, 1);
    POP(cb_scr, 1);
    POP(cb_ab, 1);
    POP(cb_dtb, 1);
    POP(cb_nega, 1);

    WAIT(cb_decay_full, 1);
    extract_scalar(cb_decay_full, cb_decay_s, b, vh);
    POP(cb_decay_full, 1);
    WAIT(cb_beta_full, 1);
    extract_scalar(cb_beta_full, cb_beta_s, b, NV + vh);
    POP(cb_beta_full, 1);

    // ---- L2 norms: qn = q * rsqrt(sum(q^2)+eps) * scale ; kn without scale ------
    WAIT(cb_qin, DKT);
    square(cb_qin, cb_sq, DKT);
    WAIT(cb_sq, DKT);
    rowsum_rsqrt(cb_sq, cb_colscale, DKT, 0, EPS, SCALE);
    POP(cb_sq, DKT);
    WAIT(cb_colscale, 1);
    bcols_mul(cb_qin, cb_colscale, cb_qn, DKT);
    POP(cb_colscale, 1);
    POP(cb_qin, DKT);

    WAIT(cb_kin, DKT);
    square(cb_kin, cb_sq, DKT);
    WAIT(cb_sq, DKT);
    rowsum_rsqrt(cb_sq, cb_colscale, DKT, 0, EPS, 0);
    POP(cb_sq, DKT);
    WAIT(cb_colscale, 1);
    bcols_mul(cb_kin, cb_colscale, cb_kn, DKT);
    POP(cb_colscale, 1);
    POP(cb_kin, DKT);

    // ---- decayed state ----------------------------------------------------------
    constexpr uint32_t KV = DKT * DVT;
    WAIT(cb_h, KV);
    WAIT(cb_decay_s, 1);
    bscalar_mul(cb_h, cb_decay_s, cb_hd, KV);
    POP(cb_h, KV);
    POP(cb_decay_s, 1);

    // ---- delta = (v - kn @ hd) * beta -------------------------------------------
    WAIT(cb_kn, DKT);
    WAIT(cb_hd, KV);
    mm_row_state(cb_kn, cb_hd, cb_hd, cb_vread, DKT, DVT);
    WAIT(cb_vin, DVT);
    WAIT(cb_vread, DVT);
    ew(cb_vin, cb_vread, cb_delta, DVT, 1);
    POP(cb_vin, DVT);
    POP(cb_vread, DVT);
    WAIT(cb_delta, DVT);
    WAIT(cb_beta_s, 1);
    bscalar_mul(cb_delta, cb_beta_s, cb_dm, DVT);
    POP(cb_delta, DVT);
    POP(cb_beta_s, 1);

    // ---- outer = kn_b^T (x) delta_b ---------------------------------------------
    // Column-broadcast form of k row b: rows of scr = kn[b, :] per block, transposed so
    // every column holds kn_b down the rows.
    cb_reserve_back(cb_scr, DKT);
    for (uint32_t m = 0; m < DKT; m++) {
        brow_mul_one(cb_ones, cb_kn, 0, m, cb_scr, m, b);
    }
    cb_push_back(cb_scr, DKT);
    POP(cb_kn, DKT);

    WAIT(cb_scr, DKT);
    cb_reserve_back(cb_kcb, DKT);
    pack_reconfig_data_format(cb_kcb);
    reconfig_data_format_srca(cb_scr);
    transpose_init(cb_scr);
    for (uint32_t m = 0; m < DKT; m++) {
        tile_regs_acquire();
        transpose_tile(cb_scr, m, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_kcb, m);
        tile_regs_release();
    }
    cb_push_back(cb_kcb, DKT);
    POP(cb_scr, DKT);

    WAIT(cb_kcb, DKT);
    WAIT(cb_dm, DVT);
    cb_reserve_back(cb_outer, KV);
    for (uint32_t m = 0; m < DKT; m++) {
        for (uint32_t n = 0; n < DVT; n++) {
            brow_mul_one(cb_kcb, cb_dm, m, n, cb_outer, m * DVT + n, b);
        }
    }
    cb_push_back(cb_outer, KV);
    POP(cb_kcb, DKT);
    POP(cb_dm, DVT);

    // ---- h_new = hd + outer (state writeback via cb_hnew) ------------------------
    WAIT(cb_outer, KV);
    ew(cb_hd, cb_outer, cb_hnew, KV, 0);

    // ---- o = qn @ hd + qn @ outer (== qn @ h_new, accumulated in dest) -----------
    WAIT(cb_qn, DKT);
    mm_row_state(cb_qn, cb_hd, cb_outer, cb_o, DKT, DVT);
    POP(cb_qn, DKT);
    POP(cb_hd, KV);
    POP(cb_outer, KV);

    // ---- gated rmsnorm: out = rmsnorm(o) * norm_w * silu(z) -----------------------
    WAIT(cb_o, DVT);
    square(cb_o, cb_sq, DVT);
    WAIT(cb_sq, DVT);
    rowsum_rsqrt(cb_sq, cb_colscale, DVT, INV_DV, NORM_EPS, 0);
    POP(cb_sq, DVT);
    WAIT(cb_colscale, 1);
    bcols_mul(cb_o, cb_colscale, cb_delta, DVT);  // cb_delta reused: normed o
    POP(cb_colscale, 1);
    POP(cb_o, DVT);

    WAIT(cb_delta, DVT);
    WAIT(cb_w, DVT);
    cb_reserve_back(cb_dm, DVT);  // cb_dm reused: normed o * norm_w
    pack_reconfig_data_format(cb_dm);
    reconfig_data_format(cb_delta, cb_w);
    mul_bcast_rows_init(cb_delta, cb_w);
    for (uint32_t n = 0; n < DVT; n++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_delta, cb_w, n, n, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_dm, n);
        tile_regs_release();
    }
    cb_push_back(cb_dm, DVT);
    POP(cb_delta, DVT);
    POP(cb_w, DVT);

    WAIT(cb_zin, DVT);
    cb_reserve_back(cb_vread, DVT);  // cb_vread reused: silu(z)
    pack_reconfig_data_format(cb_vread);
    reconfig_data_format_srca(cb_zin);
    copy_tile_init(cb_zin);
    silu_tile_init();
    for (uint32_t n = 0; n < DVT; n++) {
        tile_regs_acquire();
        copy_tile(cb_zin, n, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_vread, n);
        tile_regs_release();
    }
    cb_push_back(cb_vread, DVT);
    POP(cb_zin, DVT);

    WAIT(cb_dm, DVT);
    WAIT(cb_vread, DVT);
    ew(cb_dm, cb_vread, cb_out, DVT, 2);
    POP(cb_dm, DVT);
    POP(cb_vread, DVT);
    POP(cb_ones, 1);
}
