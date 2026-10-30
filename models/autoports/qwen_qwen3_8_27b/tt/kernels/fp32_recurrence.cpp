// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decoder-local FP32 repair of upstream fused recurrence.
// Compute kernel: the sequential-over-token recurrence for one head. Carries the recurrent state
// S [K,V] on-core across T tokens. Derived from flash-linear-attention
// `naive_recurrent_gated_delta_rule`; the vLLM `fused_sigmoid_gating_delta_rule_update` is this
// same recurrence over the K+1 speculative tokens.
//
// Per token (q pre-scaled + L2-normed, k L2-normed, decay = exp(g_t), prepared with TTNN device operations):
//   sd     = S * decay            (decay the state BEFORE the read, matches FLA + chunk scan)
//   vread  = k . sd               ([1,V])
//   u      = beta * (v - vread)   ([1,V])
//   S_new  = sd + k^T (x) u       ([K,V] rank-1 update)
//   o      = q . S_new            (read from the POST-update state)
//
// Per-token tiles are [1, D] (token in row 0, rows 1..31 zero-padded on device), so the outer-product
// update k^T (x) u is a matmul with inner dim 1 whose 31 padding lanes are zero and vanish.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
constexpr uint32_t cb_out = 6, cb_state = 7, cb_s2 = 8, cb_s3 = 9, cb_sd = 10;
constexpr uint32_t cb_vread = 11, cb_u = 12, cb_kcol = 13;
// Distinct packed copies for SrcA/SrcB consumers. The FP32 direct-to-DST
// state CBs must never be passed to matmul/transpose/FPU binary operations.
constexpr uint32_t cb_state_matrix = 14, cb_sd_matrix = 15;

// fp32 DST tiles available per tile_regs_acquire (half-sync).
constexpr uint32_t DST_TILES = 4;

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must already be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Mt * Nt);
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
    cb_o.push_back(Mt * Nt);
}

// S_new[Kt,Vt] = sd + kcol[Kt,1] (x) u[1,Vt]. Each DST tile is seeded with sd and the rank-1
// matmul accumulates onto it, so the outer product never round-trips through L1. When
// emit_state, the same DST tiles are also packed as the state output.
template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t u, uint32_t o, bool emit_state) {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_o(o);
    CircularBuffer cb_st(cb_state);
    CircularBuffer cb_matrix(cb_state_matrix);
    cb_o.reserve_back(kv);
    cb_matrix.reserve_back(kv);
    if (emit_state) {
        cb_st.reserve_back(kv);
    }
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
                pack_tile<true>(j, cb_state_matrix, mi * Vt + n0 + j);
                if (emit_state) {
                    pack_tile<true>(j, cb_state, mi * Vt + n0 + j);
                }
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(kv);
    cb_matrix.push_back(kv);
    if (emit_state) {
        cb_st.push_back(kv);
    }
}

// out = (A - B) * beta, n tiles, without spilling A - B to L1.
// All three inputs are copied directly to FP32 DST; beta scales only row 0
// because all other value rows are padding zeros.
// Preserve FP32 operands: FPU subtraction truncates source operands to TF32.
void delta_rule_residual(uint32_t a, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_init(a);
        copy_tile(a, i, 0);
        copy_init(b);
        copy_tile(b, i, 1);
        copy_init(beta);
        copy_tile(beta, 0, 2);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// out = A * scalar, n tiles. scalar is the [0,0] element of the single `scal` tile.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    CircularBuffer cb_matrix(cb_sd_matrix);
    cb_matrix.reserve_back(n);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_init(a);
        copy_tile(a, i, 0);
        copy_init(scal);
        copy_tile(scal, 0, 1);
        fill_tile_init();
        fill_tile(2, 0.0f);
        // (state - 0) * decay, entirely in FP32 DST. The scalar broadcast
        // reaches every state element, unlike a column broadcast of row 0.
        sfpu_normalize_bcast_scalar_init();
        sfpu_normalize_bcast_scalar(0, 2, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, o, i);
        pack_tile<true>(0, cb_sd_matrix, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
    cb_matrix.push_back(n);
}

// out[Kt,1] = transpose of in[1,Kt]: transpose each of the Kt tiles. (in must be available.)
void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t per_token = get_compile_time_arg_val(2);
    const uint32_t T = get_arg_val<uint32_t>(0);

    constexpr uint32_t kv = Kt * Vt;

    // FPU operands use default SrcA/SrcB formats; copy-only state/scalar CBs
    // use UnpackToDestFp32. Each init below selects the appropriate CB format.
    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    CircularBuffer q(cb_q), k(cb_k), v(cb_v), decay(cb_decay), beta(cb_beta);
    CircularBuffer sd(cb_sd), vread(cb_vread), u(cb_u), kcol(cb_kcol);
    CircularBuffer sd_matrix(cb_sd_matrix), state_matrix(cb_state_matrix);

    for (uint32_t t = 0; t < T; t++) {
        const uint32_t cur_S_id = (t == 0) ? cb_S : ((t & 1u) ? cb_s2 : cb_s3);
        const uint32_t nxt_S_id = (t & 1u) ? cb_s3 : cb_s2;
        CircularBuffer cur_S(cur_S_id), nxt_S(nxt_S_id);
        const bool last = (t == T - 1);

        // sd = cur_S * decay  (decay before read)
        decay.wait_front(1);
        cur_S.wait_front(kv);
        bcast_scalar_mul(cur_S_id, cb_decay, cb_sd, kv);
        decay.pop_front(1);
        cur_S.pop_front(kv);
        sd.wait_front(kv);

        // vread = k . sd  ([1,V])
        k.wait_front(Kt);
        sd_matrix.wait_front(kv);
        mm(cb_k, cb_sd_matrix, cb_vread, 1, Kt, Vt, false);
        sd_matrix.pop_front(kv);
        vread.wait_front(Vt);

        // u = beta * (v - vread)
        v.wait_front(Vt);
        beta.wait_front(1);
        delta_rule_residual(cb_v, cb_vread, cb_beta, cb_u, Vt);
        v.pop_front(Vt);
        vread.pop_front(Vt);
        beta.pop_front(1);
        u.wait_front(Vt);

        // kcol = transpose(k) ([K,1])
        transpose_block(cb_k, cb_kcol, Kt);
        k.pop_front(Kt);
        kcol.wait_front(Kt);

        // S_new = sd + kcol (x) u -> nxt_S (and the state output, per token or last)
        rank1_update<Kt, Vt>(cb_sd, cb_kcol, cb_u, nxt_S_id, per_token || last);
        kcol.pop_front(Kt);
        u.pop_front(Vt);
        sd.pop_front(kv);
        nxt_S.wait_front(kv);

        // o = q . S_new  (read from POST-update state)
        q.wait_front(Kt);
        state_matrix.wait_front(kv);
        mm(cb_q, cb_state_matrix, cb_out, 1, Kt, Vt, false);
        state_matrix.pop_front(kv);
        q.pop_front(Kt);

        // nxt_S is intentionally NOT popped: the next iteration reads it as cur_S.
    }
}
