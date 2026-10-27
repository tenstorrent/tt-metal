// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Prep compute: the state-independent half of the chunked gated delta rule forward, one
// (value head, chunk) work-item at a time; the recurrent state lives in the scan program.
// Derived from flash-linear-attention `naive_chunk_gated_delta_rule`. fp32 CBs and HiFi4; FPU
// operands are read through the ~tf32 SrcA/SrcB registers.
//
// Per chunk (C=chunk, K=key dim, V=val dim; Ct=C/32, Kt=K/32, Vt=V/32), with q already scaled and
// q, k already L2-normalized along K by the caller:
//   v_beta = v*beta ; k_beta = k*beta
//   decay = cumsum(g) = tril @ g ; decay_exp = exp(decay)          (decay is teed out as g_cumsum)
//   L_mask = tril( exp(decay_i - decay_j) )
//   N = strictly_lower(k_beta@k^T * L_mask) ; T_inv = (I+N)^-1     (T_inv is the saved A)
//   kd = k_beta*decay_exp ; intra = (q@k^T) * L_mask
//   q_decay = q*decay_exp ; k_dec_t = transpose(k * exp(decay_last - decay)) ; dl = exp(decay_last)

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"
#include "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_fw/device/kernels/chunk_gated_delta_rule_fw_common.hpp"

namespace {

using namespace chunk_gated_delta_rule_fw::prep;

// Scratch-CB push quantum. cb_scr1/2/3 carry blocks of many sizes (1, Ct, cc, ck, kc tiles). The
// LLK pack/unpack address of tile i is fifo_ptr + i*page with no wrap
// (llk_pack_common_api.h::get_output_tile_address), so a block that straddles the ring end would
// read/write past the CB into the next CB's L1. Every reserve/push/wait/pop on a scratch CB therefore
// moves the whole ring (SCR tiles, its capacity): each block starts at the ring base and indexes
// tiles 0..n-1. This requires each scratch CB to hold at most one live block.
constexpr uint32_t kCt = get_compile_time_arg_val(0);
constexpr uint32_t kKt = get_compile_time_arg_val(1);
constexpr uint32_t kVt = get_compile_time_arg_val(2);
constexpr uint32_t cmax(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}
constexpr uint32_t SCR = cmax(cmax(cmax(kCt * kCt, kCt* kKt), cmax(kCt* kVt, kKt* kVt)), kKt* kCt);
inline uint32_t qn(uint32_t cb, uint32_t n) {
    return (cb == cb_scr1 || cb == cb_scr2 || cb == cb_scr3) ? SCR : n;
}

inline void WAIT(uint32_t cb, uint32_t n) {
    CircularBuffer(cb).wait_front(qn(cb, n));
}
inline void POP(uint32_t cb, uint32_t n) {
    CircularBuffer(cb).pop_front(qn(cb, n));
}

constexpr uint32_t NO_CB = 0xFFFFFFFFu;

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
// o2 (optional, same data format as o): every result tile is ALSO packed from DEST into o2 — an
// exact fp32 tee for a writer-drained output (a copy_tile round-trip would unpack through the tf32
// source register and truncate it).
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr, uint32_t o2 = NO_CB) {
    cb_reserve_back(o, qn(o, Mt * Nt));
    if (o2 != NO_CB) {
        cb_reserve_back(o2, qn(o2, Mt * Nt));
    }
    pack_reconfig_data_format(o);  // mixed bf16/fp32 CBs: set packer to this output's format
    // matmul_tiles(a,b): in0=a->srcB, in1=b->srcA. Reconfig unpack src formats to match (the op
    // init only asserts formats, it does not set them), else fp32/bf16 CBs are read at the wrong
    // format and produce garbage.
    reconfig_data_format(b, a);
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
            if (o2 != NO_CB) {
                pack_tile(0, o2, mi * Nt + ni);
            }
            tile_regs_release();
        }
    }
    cb_push_back(o, qn(o, Mt * Nt));
    if (o2 != NO_CB) {
        cb_push_back(o2, qn(o2, Mt * Nt));
    }
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, qn(o, n));
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
    cb_push_back(o, qn(o, n));
}

void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, qn(o, n));
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    copy_init(in);
    exp_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, qn(o, n));
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1]  (broadcast the single column of `col` across N)
void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, qn(o, Mt * Nt));
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(a, col);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, qn(o, Mt * Nt));
}

// out[Mt,Nt] = A[Mt,Nt] - row[1,Nt]  (broadcast the single row of `row` across M)
void bcast_rows_sub(uint32_t a, uint32_t row, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, qn(o, Mt * Nt));
    pack_reconfig_data_format(o);
    reconfig_data_format(a, row);  // bcast(a,row): a->srcA, row->srcB
    sub_bcast_rows_init(a, row);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            sub_tiles_bcast_rows(a, row, mi * Nt + ni, ni, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, qn(o, Mt * Nt));
}

// out[0] = copy of src[src_tile] (single 32x32 tile). src must be available.
void cpy_t(uint32_t src, uint32_t src_tile, uint32_t o) {
    cb_reserve_back(o, qn(o, 1));
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(src);
    copy_init(src);
    tile_regs_acquire();
    copy_tile(src, src_tile, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, qn(o, 1));
}

// out[0] = a[ai] (op) b[bi], single tile. op: 0 add, 2 mul. (Like ew but with free tile indices.)
void ewt(uint32_t a, uint32_t ai, uint32_t b, uint32_t bi, uint32_t o, int op) {
    cb_reserve_back(o, qn(o, 1));
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (op == 0) {
        add_init(a, b);
    } else {
        mul_init(a, b);
    }
    tile_regs_acquire();
    if (op == 0) {
        add_tiles(a, b, ai, bi, 0);
    } else {
        mul_tiles(a, b, ai, bi, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, qn(o, 1));
}

// (I32 - Nq)^-1 for a strictly-lower 16-block Nq isolated in one 16-quadrant (rest zero),
// nilpotent at 16: Horner over Nq^0..Nq^15 -> out (single tile); the other diagonal quadrant is I.
// Small block + short chain keeps fp32 bounded where a 32x32/31-term Horner cancels.
void invert16(uint32_t nq, uint32_t out, uint32_t tmp) {
    ew(cb_eye, nq, out, 1, 0);  // out = I + Nq
    WAIT(out, 1);
    for (uint32_t m = 2; m < 16; m++) {  // sum_{k<16} Nq^k
        mm(nq, out, tmp, 1, 1, 1, false);
        WAIT(tmp, 1);
        POP(out, 1);
        ew(cb_eye, tmp, out, 1, 0);  // out = I + Nq @ out
        WAIT(out, 1);
        POP(tmp, 1);
    }
}

// Assemble the 2x2 tile-block matrix [[s0[t0], s1[t1]], [s2[t2], s3[t3]]] into o (4 tiles).
void asm4(
    uint32_t s0,
    uint32_t t0,
    uint32_t s1,
    uint32_t t1,
    uint32_t s2,
    uint32_t t2,
    uint32_t s3,
    uint32_t t3,
    uint32_t o) {
    const uint32_t src[4] = {s0, s1, s2, s3};
    const uint32_t tl[4] = {t0, t1, t2, t3};
    cb_reserve_back(o, qn(o, 4));
    pack_reconfig_data_format(o);
    for (uint32_t i = 0; i < 4; i++) {
        reconfig_data_format_srca(src[i]);
        copy_init(src[i]);
        tile_regs_acquire();
        copy_tile(src[i], tl[i], 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, qn(o, 4));
}

// Invert one 32x32 diagonal tile-block: out[0] = (I32 - negN)^-1, negN = src[tile] (strictly-lower
// 32x32). Mirrors FLA solve_tril: split into 16-quadrants negN = [[N00,0],[N10,N11]], invert the two
// diagonal 16-blocks (short, bounded Horners), and form the off-diagonal EXACTLY (one matmul chain,
// no power series). A single 32x32 Horner instead loses fp32 precision on harder blocks.
//   Bi00=(I-N00)^-1 (top-left), Bi11=(I-N11)^-1 (bottom-right), off=Bi11@N10@Bi00 (bottom-left).
//   out = [[Bi00,0],[off,Bi11]].
// Masks (single tiles): cb_mask[0]=Qtl, [1]=Qbr, [2]=Q10 (bottom-left). tmpN/tmpT = scratch.
// Private scratch A..D use cb_S/cb_final/cb_s2/cb_s3 — single-tile fp32 CBs NOT drained by the prep
// writer (unlike the output CBs cb_kd/cb_qdecay/cb_intra, whose scratch pushes the writer would
// wrongly consume). None alias src (cb_scr3), out, or the Ct==2 persistents (cb_supd/cb_stmp).
void invert_block(uint32_t src, uint32_t tile, uint32_t out, uint32_t tmpN, uint32_t tmpT) {
    const uint32_t A = cb_S, B = cb_final, C = cb_s2, D = cb_s3;
    cpy_t(src, tile, tmpN);
    WAIT(tmpN, 1);  // negN -> tmpN[0]
    // Bi00 = (I-N00)^-1  (N00 = top-left quadrant of negN; top-right is already 0)
    ewt(tmpN, 0, cb_mask, 0, A, 2);
    WAIT(A, 1);  // N00
    invert16(A, B, tmpT);
    WAIT(B, 1);
    POP(A, 1);  // Bi00 -> B
    // Bi11 = (I-N11)^-1  (N11 = bottom-right quadrant)
    ewt(tmpN, 0, cb_mask, 1, A, 2);
    WAIT(A, 1);  // N11
    invert16(A, C, tmpT);
    WAIT(C, 1);
    POP(A, 1);  // Bi11 -> C
    // off = Bi11 @ N10 @ Bi00  (N10 = bottom-left quadrant; result lives only there)
    ewt(tmpN, 0, cb_mask, 2, A, 2);
    WAIT(A, 1);  // N10
    POP(tmpN, 1);
    mm(C, A, tmpT, 1, 1, 1, false);
    WAIT(tmpT, 1);
    POP(A, 1);  // Bi11@N10
    mm(tmpT, B, A, 1, 1, 1, false);
    WAIT(A, 1);
    POP(tmpT, 1);  // @Bi00 -> A(off)
    // out = Qtl*Bi00 + Qbr*Bi11 + off
    ewt(B, 0, cb_mask, 0, D, 2);
    WAIT(D, 1);
    POP(B, 1);  // Bi00_tl -> D
    ewt(C, 0, cb_mask, 1, B, 2);
    WAIT(B, 1);
    POP(C, 1);  // Bi11_br -> B
    ewt(D, 0, B, 0, C, 0);
    WAIT(C, 1);
    POP(D, 1);
    POP(B, 1);
    ewt(C, 0, A, 0, out, 0);
    POP(C, 1);
    POP(A, 1);  // + off -> out
}

// out[1,Ct] row-form = transpose of col[Ct,1]; produces Ct tiles (each row0 = a 32-chunk of col).
void transpose_col(uint32_t in, uint32_t o, uint32_t Ct) {
    cb_reserve_back(o, qn(o, Ct));
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);  // unary: in->srcA
    transpose_init(in);
    for (uint32_t i = 0; i < Ct; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, qn(o, Ct));
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    // This core's work-item count. Items are independent (head, chunk) preps.
    const uint32_t num_items = get_arg_val<uint32_t>(0);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    static_assert(Ct == 1 || Ct == 2, "the WY inverse is assembled for chunk_size 32 or 64 only");

    compute_kernel_hw_startup(cb_q, cb_k, cb_u);

    // Constants, built once by the reader.
    WAIT(cb_eye, cc);
    WAIT(cb_tril, cc);
    WAIT(cb_ones, cc);
    WAIT(cb_mask, kNumMasks);  // Qtl, Qbr, Q10 (used by invert_block)

    // Per item: v_beta, T_inv, kd, intra, q_decay, k_dec_t, dl and g_cumsum stay pushed in their CBs
    // for the prep writer.
    for (uint32_t c = 0; c < num_items; c++) {
        WAIT(cb_q, ck);
        WAIT(cb_k, ck);
        WAIT(cb_v, cv);
        WAIT(cb_g, Ct);
        WAIT(cb_beta, Ct);

        constexpr uint32_t Q = cb_q, Kk = cb_k;

        // ---- P1: v_beta, k_beta ----
        bcast_cols_mul(cb_v, cb_beta, cb_vbeta, Ct, Vt);
        WAIT(cb_vbeta, cv);
        bcast_cols_mul(Kk, cb_beta, cb_kbeta, Ct, Kt);
        WAIT(cb_kbeta, ck);
        POP(cb_beta, Ct);
        POP(cb_v, cv);

        // ---- P2: decay = tril@g, decay_exp, decay_row ----
        // (decay is also packed straight into cb_gcum -> writer -> g_cumsum.)
        mm(cb_tril, cb_g, cb_decay, Ct, Ct, 1, false, cb_gcum);
        WAIT(cb_decay, Ct);
        expc(cb_decay, cb_decay_exp, Ct);
        WAIT(cb_decay_exp, Ct);
        transpose_col(cb_decay, cb_scr1, Ct);  // decay_row in scr1
        WAIT(cb_scr1, Ct);

        // ---- L_mask = tril(exp(decay_i - decay_j)) ----
        bcast_cols_mul(cb_ones, cb_decay, cb_scr2, Ct, Ct);  // decay_i everywhere
        WAIT(cb_scr2, cc);
        bcast_rows_sub(cb_scr2, cb_scr1, cb_scr3, Ct, Ct);  // decay_i - decay_j
        WAIT(cb_scr3, cc);
        POP(cb_scr1, Ct);  // decay_row done
        POP(cb_scr2, cc);
        ew(cb_scr3, cb_tril, cb_scr2, cc, 2);  // *tril (zero upper)
        WAIT(cb_scr2, cc);
        POP(cb_scr3, cc);
        expc(cb_scr2, cb_scr3, cc);  // exp
        WAIT(cb_scr3, cc);
        POP(cb_scr2, cc);
        ew(cb_scr3, cb_tril, cb_lmask, cc, 2);  // *tril again -> L_mask
        WAIT(cb_lmask, cc);
        POP(cb_scr3, cc);

        // ---- decayfac = exp(g_sum - decay) ----
        mm(cb_ones, cb_g, cb_scr1, Ct, Ct, 1, false);  // g_sum in every row (col form)
        WAIT(cb_scr1, Ct);
        POP(cb_g, Ct);
        ew(cb_scr1, cb_decay, cb_scr2, Ct, 1);  // g_sum - decay
        WAIT(cb_scr2, Ct);
        POP(cb_scr1, Ct);
        POP(cb_decay, Ct);
        expc(cb_scr2, cb_decayfac, Ct);
        WAIT(cb_decayfac, Ct);
        POP(cb_scr2, Ct);

        // ---- N = strictly_lower(k_beta@k^T * L_mask); T_inv = (I + strictly_lower)^-1 ----
        // The WY inverse, mirroring FLA's solve_tril: block down to 16x16 (invert_block splits each
        // 32x32 tile into 16-quadrants), invert the small diagonal blocks with bounded Horners, and
        // merge off-diagonal blocks EXACTLY. This keeps every intermediate bounded, unlike a single
        // 32x32/full-matrix Horner whose deep power series loses fp32 precision on harder chunks.
        mm(cb_kbeta, Kk, cb_scr1, Ct, Kt, Ct, true);  // kk = k_beta @ k^T
        WAIT(cb_scr1, cc);
        ew(cb_scr1, cb_lmask, cb_scr2, cc, 2);  // kk_masked = kk * L_mask
        WAIT(cb_scr2, cc);
        POP(cb_scr1, cc);
        ew(cb_scr2, cb_eye, cb_scr1, cc, 2);  // diag(kk_masked)
        WAIT(cb_scr1, cc);
        // negN = diag - kk_masked = -(strictly_lower(kk_masked))  (= -A_strict, kept in cb_scr3)
        ew(cb_scr1, cb_scr2, cb_scr3, cc, 1);
        WAIT(cb_scr3, cc);
        POP(cb_scr1, cc);
        POP(cb_scr2, cc);

        if constexpr (Ct == 1) {
            // Single 32x32 block: T_inv is just its inverse.
            invert_block(cb_scr3, 0, cb_Tinv, cb_scr1, cb_scr2);
            WAIT(cb_Tinv, cc);
            POP(cb_scr3, cc);
        } else if constexpr (Ct == 2) {
            // 2x2 tile-block lower-triangular. negN tiles: 0=(0,0), 2=(1,0), 3=(1,1); (0,1)=0.
            // Diagonal inverses Mi11, Mi22, then off-diagonal Mi21 = -Mi22 @ A21 @ Mi11.
            // (A21 = -negN21, so -Mi22@A21@Mi11 = Mi22 @ negN21 @ Mi11.)
            // Mi11 -> cb_supd, Mi22 -> cb_stmp, Mi21 -> cb_ointer (all free in prep).
            invert_block(cb_scr3, 0, cb_supd, cb_scr1, cb_scr2);  // Mi11
            invert_block(cb_scr3, 3, cb_stmp, cb_scr1, cb_scr2);  // Mi22
            cpy_t(cb_scr3, 2, cb_scr1);                           // negN21 -> cb_scr1[0]
            WAIT(cb_scr1, 1);
            mm(cb_scr1, cb_supd, cb_scr2, 1, 1, 1, false);  // tmp = negN21 @ Mi11
            WAIT(cb_scr2, 1);
            POP(cb_scr1, 1);
            mm(cb_stmp, cb_scr2, cb_ointer, 1, 1, 1, false);  // Mi21 = Mi22 @ tmp
            WAIT(cb_ointer, 1);
            POP(cb_scr2, 1);
            POP(cb_scr3, cc);  // negN done
            // T_inv = [[Mi11, 0], [Mi21, Mi22]]  (cb_eye[1] is the zero block)
            asm4(cb_supd, 0, cb_eye, 1, cb_ointer, 0, cb_stmp, 0, cb_Tinv);
            WAIT(cb_Tinv, cc);
            POP(cb_supd, 1);
            POP(cb_stmp, 1);
            POP(cb_ointer, 1);
        }

        // ---- un-premultiplied WY hand-off: output v_beta (cb_vbeta), kd=k_beta*decay_exp (cb_kd),
        // T_inv (cb_Tinv). The scan computes v_new = T_inv @ (v_beta - kd@S) instead of pre-multiplying
        // u = T_inv@v_beta, w = T_inv@kd, so the inverse's fp error is not amplified by the u - w@S
        // cancellation.
        bcast_cols_mul(cb_kbeta, cb_decay_exp, cb_kd, Ct, Kt);
        WAIT(cb_kd, ck);
        POP(cb_kbeta, ck);
        // cb_vbeta (v_beta) and cb_Tinv (T_inv) remain pushed for the writer; NOT popped here.

        // ---- intra = (q@k^T) * L_mask ; q_decay = q*decay_exp ; k_dec_t ----
        mm(Q, Kk, cb_scr1, Ct, Kt, Ct, true);  // qk = q @ k^T
        WAIT(cb_scr1, cc);
        ew(cb_scr1, cb_lmask, cb_intra, cc, 2);
        WAIT(cb_intra, cc);
        POP(cb_scr1, cc);
        POP(cb_lmask, cc);
        bcast_cols_mul(Q, cb_decay_exp, cb_qdecay, Ct, Kt);
        WAIT(cb_qdecay, ck);
        POP(Q, ck);
        // decay_exp and decayfac stay pushed until dl below.
        bcast_cols_mul(Kk, cb_decayfac, cb_scr1, Ct, Kt);  // k * exp(decay_last-decay)
        WAIT(cb_scr1, ck);
        POP(Kk, ck);
        // k_dec_t = transpose(k_dec) [K,C]: transpose each [Ct,Kt] tile block into [Kt,Ct].
        cb_reserve_back(cb_kdec_t, qn(cb_kdec_t, Kt * Ct));
        pack_reconfig_data_format(cb_kdec_t);
        reconfig_data_format_srca(cb_scr1);  // unary: in->srcA
        transpose_init(cb_scr1);
        for (uint32_t ki = 0; ki < Kt; ki++) {
            for (uint32_t ci = 0; ci < Ct; ci++) {
                tile_regs_acquire();
                transpose_tile(cb_scr1, ci * Kt + ki, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_kdec_t, ki * Ct + ci);
                tile_regs_release();
            }
        }
        cb_push_back(cb_kdec_t, qn(cb_kdec_t, Kt * Ct));
        POP(cb_scr1, ck);

        // ---- dl = exp(g_sum) = decayfac[i]*decay_exp[i] (same for all i); 1 tile, [0,0] holds dl.
        // The scan kernel uses it to decay the recurrent state: S <- S*dl + k_dec_t@v_new.
        ew(cb_decayfac, cb_decay_exp, cb_dl, 1, 2);
        WAIT(cb_dl, 1);
        POP(cb_decayfac, Ct);
        POP(cb_decay_exp, Ct);
    }
}
