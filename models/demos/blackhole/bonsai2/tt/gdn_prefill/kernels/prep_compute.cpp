// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet chunk preprocessing for gated_delta_attn_seq, compute. Per item (head row, chunk of C = 128 tokens),
// with q already scaled, everything fp32 (HiFi4 matmuls, fp32 DEST), tiles (i, j) of the 4 x 4 [C, C] grid:
//   drow = g @ triu (prefix sum, row form), dcol = its transpose, S = drow[C-1]
//   dexp = exp(clip(dcol)), kdf = exp(clip(S - dcol)), dl_exp = exp(clip(S))
//   LM_ij = exp(clip(dcol_i - drow_j)) (diagonal tiles masked to the lower triangle), zero above the diagonal
//   M = ((k * beta) @ k^T) * LM, s = diag(M), D_inv = 1 / (1 + alpha s)
//   L_unit = I + strict_lower(M) * D_inv (row scale), v_beta_sc = v * beta * D_inv, k_bd_sc = k * beta * dexp * D_inv
//   intra = (q @ k^T) * LM, q_decay = q * dexp, k_decay_t = (k * kdf)^T
//   L_inv: inverse of each 32 x 32 diagonal block of L_unit = I + N (N strictly lower, nilpotent):
//          (I - N)(I + N^2)(I + N^4)(I + N^8)(I + N^16), exact.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "../../ternary/kernels/core_span.hpp"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_b = 3, cb_g = 4, cb_c = 5;
constexpr uint32_t cb_gcol = 6, cb_bd = 7, cb_dcol = 8, cb_dexp = 9, cb_kdf = 10, cb_gb = 11, cb_lm = 12, cb_sfull = 27;
constexpr uint32_t cb_kb = 13, cb_kt = 14, cb_m = 15, cb_out = 16, cb_scol = 17, cb_dinv = 18, cb_sl = 19;
constexpr uint32_t cb_w1 = 20, cb_w2 = 21, cb_p = 23, cb_r = 24, cb_t = 25;
constexpr uint32_t cb_tmp = 26;
constexpr uint32_t cb_kd = cb_kb;  // k * beta is done by then
// constant tiles in cb_c
constexpr uint32_t TRIU = 0, ONES = 1, TRIL = 2, STRIL = 3, EYE = 4, E31 = 5, ZERO = 6, STRIU = 7;
constexpr uint32_t NEG20 = 0xC1A00000u, ZERO_F = 0u, ONE_F = 0x3F800000u;
constexpr auto TO_A = EltwiseBinaryReuseDestType::DEST_TO_SRCA;

ALWI uint32_t lidx(uint32_t i, uint32_t j) { return i * (i + 1) / 2 + j; }  // lower-triangle tile (i >= j)

// Pack DEST tiles 0..n-1 into cb and push. cb_out takes them one page at a time: its puts mix 1 and 8 tiles, and an
// out-of-order pack of n tiles at once must not run past the end of the ring.
ALWI void put(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    tile_regs_wait();
    if (cb == cb_out) {
        for (uint32_t d = 0; d < n; ++d) {
            cb_reserve_back(cb, 1);
            pack_tile<true>(d, cb, 0);
            cb_push_back(cb, 1);
        }
    } else {
        cb_reserve_back(cb, n);
        for (uint32_t d = 0; d < n; ++d) {
            pack_tile<true>(d, cb, d);
        }
        cb_push_back(cb, n);
    }
    tile_regs_release();
}
ALWI void clip_exp(uint32_t d) {
    clamp_tile_init();
    clamp_tile(d, NEG20, ZERO_F);
    exp_tile_init();
    exp_tile(d);
}
ALWI void zero_out() {  // one zero tile to cb_out
    tile_regs_acquire();
    copy_init(cb_c);
    copy_tile(cb_c, ZERO, 0);
    put(cb_out, 1);
}
}  // namespace

void kernel_main() {
    // common args: Nitems, ncores, grid_x, alpha (fp32 bits)
    uint32_t first, n;
    core_span(
        get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), 1, first, n);
    const uint32_t alpha = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t NCONST = 8;

    compute_kernel_hw_startup(cb_q, cb_k, cb_out);
    cb_wait_front(cb_c, NCONST);

    for (uint32_t it = 0; it < n; ++it) {
        cb_wait_front(cb_g, 4);
        cb_wait_front(cb_b, 4);
        cb_wait_front(cb_k, 16);
        cb_wait_front(cb_q, 16);

        // No prefix sum is ever subtracted from another: the FPU rounds fp32 inputs to tf32, and decay_i - decay_j of
        // two sums near -1000 lost the result (6% error on real decays). Every difference is its own partial sum.
        // gcol_t = g_t^T (column tiles); GB_t = gcol_t across columns; BD_t = GB_t * STRIL
        tile_regs_acquire();
        transpose_init(cb_g);
        for (uint32_t t = 0; t < 4; ++t) {
            transpose_tile(cb_g, t, t);
        }
        put(cb_gcol, 4);
        cb_wait_front(cb_gcol, 4);
        tile_regs_acquire();
        unary_bcast_init<BroadcastType::COL>(cb_gcol);
        for (uint32_t t = 0; t < 4; ++t) {
            unary_bcast<BroadcastType::COL>(cb_gcol, t, t);
        }
        unary_bcast_uninit<BroadcastType::COL>(cb_gcol);
        put(cb_gb, 4);
        cb_wait_front(cb_gb, 4);
        tile_regs_acquire();
        mul_init(cb_gb, cb_c);
        for (uint32_t t = 0; t < 4; ++t) {
            mul_tiles(cb_gb, cb_c, t, STRIL, t);
        }
        put(cb_bd, 4);
        cb_wait_front(cb_bd, 4);
        // dcol_i = sum_{t<i} ONES @ gcol_t + TRIL @ gcol_i (prefix sums, column form); sfull = sum of all g
        tile_regs_acquire();
        matmul_init(cb_c, cb_gcol);
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t t = 0; t < i; ++t) {
                matmul_tiles(cb_c, cb_gcol, ONES, t, i);
            }
            matmul_tiles(cb_c, cb_gcol, TRIL, i, i);
        }
        put(cb_dcol, 4);
        cb_wait_front(cb_dcol, 4);
        tile_regs_acquire();
        matmul_init(cb_c, cb_gcol);
        for (uint32_t t = 0; t < 4; ++t) {
            matmul_tiles(cb_c, cb_gcol, ONES, t, 0);
        }
        put(cb_sfull, 1);
        cb_wait_front(cb_sfull, 1);
        // dexp
        tile_regs_acquire();
        copy_init(cb_dcol);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_dcol, i, i);
        }
        for (uint32_t i = 0; i < 4; ++i) {
            clip_exp(i);
        }
        put(cb_dexp, 4);
        // kdf_i = exp(clip(sum of g after each row)) = exp(clip(STRIU @ gcol_i + sum_{t>i} ONES @ gcol_t))
        tile_regs_acquire();
        matmul_init(cb_c, cb_gcol);
        for (uint32_t i = 0; i < 4; ++i) {
            matmul_tiles(cb_c, cb_gcol, STRIU, i, i);
            for (uint32_t t = i + 1; t < 4; ++t) {
                matmul_tiles(cb_c, cb_gcol, ONES, t, i);
            }
        }
        for (uint32_t i = 0; i < 4; ++i) {
            clip_exp(i);
        }
        put(cb_kdf, 4);
        // LM_ij = exp(clip(sum_{t=j}^{i} A_it @ B_tj)) with A_ii = TRIL, A_it = ONES (t < i), B_jj = BD_j,
        // B_tj = GB_t (t > j): entry (r, c) is the sum of g over (c, r]. Diagonal tiles masked to the lower triangle.
        cb_reserve_back(cb_lm, 10);
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t j = 0; j <= i; ++j) {
                tile_regs_acquire();
                matmul_init(cb_c, cb_bd);
                matmul_tiles(cb_c, cb_bd, i == j ? TRIL : ONES, j, 0);
                if (i > j) {
                    matmul_init(cb_c, cb_gb);
                    for (uint32_t t = j + 1; t <= i; ++t) {
                        matmul_tiles(cb_c, cb_gb, t == i ? TRIL : ONES, t, 0);
                    }
                }
                clip_exp(0);
                if (i == j) {
                    mul_reuse_dest_init<TO_A>(cb_c);
                    mul_reuse_dest_tiles<TO_A>(cb_c, TRIL, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_lm, lidx(i, j));
                tile_regs_release();
            }
        }
        cb_push_back(cb_lm, 10);
        cb_wait_front(cb_lm, 10);
        cb_pop_front(cb_gb, 4);
        cb_pop_front(cb_bd, 4);
        // kb = k * beta (row scale); kt = k^T
        for (uint32_t h = 0; h < 2; ++h) {
            tile_regs_acquire();
            mul_bcast_cols_init(cb_k, cb_b);
            for (uint32_t x = 0; x < 8; ++x) {
                const uint32_t idx = h * 8 + x;
                mul_tiles_bcast_cols(cb_k, cb_b, idx, idx / 4, x);
            }
            put(cb_kb, 8);
        }
        cb_wait_front(cb_kb, 16);
        for (uint32_t h = 0; h < 2; ++h) {
            tile_regs_acquire();
            transpose_init(cb_k);
            for (uint32_t x = 0; x < 8; ++x) {
                const uint32_t idx = h * 8 + x, t = idx / 4, j = idx % 4;  // kt tile (t, j) = (k tile (j, t))^T
                transpose_tile(cb_k, j * 4 + t, x);
            }
            put(cb_kt, 8);
        }
        cb_wait_front(cb_kt, 16);
        // M = (kb @ kt) * LM, lower tiles
        cb_reserve_back(cb_m, 10);
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t j = 0; j <= i; ++j) {
                tile_regs_acquire();
                matmul_init(cb_kb, cb_kt);
                for (uint32_t t = 0; t < 4; ++t) {
                    matmul_tiles(cb_kb, cb_kt, i * 4 + t, t * 4 + j, 0);
                }
                mul_reuse_dest_init<TO_A>(cb_lm);
                mul_reuse_dest_tiles<TO_A>(cb_lm, lidx(i, j), 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_m, lidx(i, j));
                tile_regs_release();
            }
        }
        cb_push_back(cb_m, 10);
        cb_wait_front(cb_m, 10);
        cb_pop_front(cb_kb, 16);
        // s_i (all columns) = (M_ii * EYE) @ ONES; D_inv = 1 / (1 + alpha s)
        tile_regs_acquire();
        mul_init(cb_m, cb_c);
        for (uint32_t i = 0; i < 4; ++i) {
            mul_tiles(cb_m, cb_c, lidx(i, i), EYE, i);
        }
        put(cb_tmp, 4);
        cb_wait_front(cb_tmp, 4);
        tile_regs_acquire();
        matmul_init(cb_tmp, cb_c);
        for (uint32_t i = 0; i < 4; ++i) {
            matmul_tiles(cb_tmp, cb_c, i, ONES, i);
        }
        binop_with_scalar_tile_init();
        for (uint32_t i = 0; i < 4; ++i) {
            mul_unary_tile(i, alpha);
            add_unary_tile(i, ONE_F);
        }
        recip_tile_init();
        for (uint32_t i = 0; i < 4; ++i) {
            recip_tile(i);
        }
        put(cb_dinv, 4);
        cb_pop_front(cb_tmp, 4);
        cb_wait_front(cb_dinv, 4);
        // L_unit (out) and SL_i = strict_lower(M_ii) * D_inv_i (for the block inverses)
        cb_reserve_back(cb_sl, 4);
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t j = 0; j < 4; ++j) {
                if (j > i) {
                    zero_out();
                    continue;
                }
                tile_regs_acquire();
                mul_bcast_cols_init(cb_m, cb_dinv);
                mul_tiles_bcast_cols(cb_m, cb_dinv, lidx(i, j), i, 0);
                if (i == j) {
                    mul_tiles_bcast_cols(cb_m, cb_dinv, lidx(i, j), i, 1);
                    mul_reuse_dest_init<TO_A>(cb_c);
                    mul_reuse_dest_tiles<TO_A>(cb_c, STRIL, 0);
                    mul_reuse_dest_tiles<TO_A>(cb_c, STRIL, 1);
                    add_reuse_dest_init<TO_A>(cb_c);
                    add_reuse_dest_tiles<TO_A>(cb_c, EYE, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_out, 1);
                pack_tile<true>(0, cb_out, 0);
                cb_push_back(cb_out, 1);
                if (i == j) {
                    pack_tile<true>(1, cb_sl, i);
                }
                tile_regs_release();
            }
        }
        cb_push_back(cb_sl, 4);
        cb_pop_front(cb_m, 10);
        // w1 = beta * D_inv, w2 = w1 * dexp (column vectors)
        tile_regs_acquire();
        mul_init(cb_b, cb_dinv);
        for (uint32_t i = 0; i < 4; ++i) {
            mul_tiles(cb_b, cb_dinv, i, i, i);
        }
        put(cb_w1, 4);
        cb_wait_front(cb_w1, 4);
        cb_wait_front(cb_dexp, 4);
        tile_regs_acquire();
        mul_init(cb_w1, cb_dexp);
        for (uint32_t i = 0; i < 4; ++i) {
            mul_tiles(cb_w1, cb_dexp, i, i, i);
        }
        put(cb_w2, 4);
        cb_wait_front(cb_w2, 4);
        // v_beta_sc = v * w1, k_bd_sc = k * w2
        for (uint32_t h = 0; h < 2; ++h) {  // v arrives in two halves of 8 tiles
            cb_wait_front(cb_v, 8);
            tile_regs_acquire();
            mul_bcast_cols_init(cb_v, cb_w1);
            for (uint32_t x = 0; x < 8; ++x) {
                mul_tiles_bcast_cols(cb_v, cb_w1, x, (h * 8 + x) / 4, x);
            }
            put(cb_out, 8);
            cb_pop_front(cb_v, 8);
        }
        for (uint32_t h = 0; h < 2; ++h) {
            tile_regs_acquire();
            mul_bcast_cols_init(cb_k, cb_w2);
            for (uint32_t x = 0; x < 8; ++x) {
                const uint32_t idx = h * 8 + x;
                mul_tiles_bcast_cols(cb_k, cb_w2, idx, idx / 4, x);
            }
            put(cb_out, 8);
        }
        // intra = (q @ kt) * LM
        for (uint32_t i = 0; i < 4; ++i) {
            for (uint32_t j = 0; j < 4; ++j) {
                if (j > i) {
                    zero_out();
                    continue;
                }
                tile_regs_acquire();
                matmul_init(cb_q, cb_kt);
                for (uint32_t t = 0; t < 4; ++t) {
                    matmul_tiles(cb_q, cb_kt, i * 4 + t, t * 4 + j, 0);
                }
                mul_reuse_dest_init<TO_A>(cb_lm);
                mul_reuse_dest_tiles<TO_A>(cb_lm, lidx(i, j), 0);
                put(cb_out, 1);
            }
        }
        // q_decay = q * dexp
        for (uint32_t h = 0; h < 2; ++h) {
            tile_regs_acquire();
            mul_bcast_cols_init(cb_q, cb_dexp);
            for (uint32_t x = 0; x < 8; ++x) {
                const uint32_t idx = h * 8 + x;
                mul_tiles_bcast_cols(cb_q, cb_dexp, idx, idx / 4, x);
            }
            put(cb_out, 8);
        }
        // k_decay_t: kd = k * kdf, then tile (t, i) of the output = (kd tile (i, t))^T
        cb_wait_front(cb_kdf, 4);
        for (uint32_t h = 0; h < 2; ++h) {
            tile_regs_acquire();
            mul_bcast_cols_init(cb_k, cb_kdf);
            for (uint32_t x = 0; x < 8; ++x) {
                const uint32_t idx = h * 8 + x;
                mul_tiles_bcast_cols(cb_k, cb_kdf, idx, idx / 4, x);
            }
            put(cb_kd, 8);
        }
        cb_wait_front(cb_kd, 16);
        for (uint32_t h = 0; h < 2; ++h) {
            tile_regs_acquire();
            transpose_init(cb_kd);
            for (uint32_t x = 0; x < 8; ++x) {
                const uint32_t o = h * 8 + x, t = o / 4, i = o % 4;
                transpose_tile(cb_kd, i * 4 + t, x);
            }
            put(cb_out, 8);
        }
        cb_pop_front(cb_kd, 16);
        // dl_exp
        tile_regs_acquire();
        copy_init(cb_sfull);
        copy_tile(cb_sfull, 0, 0);
        clip_exp(0);
        put(cb_out, 1);
        // L_inv blocks by Horner, R <- I + (-N) R from R = I, 31 steps (N^32 = 0): R stays near the inverse at every
        // step. The exact product (I - N)(I + N^2)...(I + N^16) is unstable here: correlated keys put N's entries near
        // -1, N^16 reaches ~1e8 and the product cancels it (289x error on real layer-0 inputs).
        cb_wait_front(cb_sl, 4);
        tile_regs_acquire();
        copy_init(cb_sl);
        for (uint32_t b = 0; b < 4; ++b) {
            copy_tile(cb_sl, b, b);
        }
        negative_tile_init();
        for (uint32_t b = 0; b < 4; ++b) {
            negative_tile(b);
        }
        put(cb_p, 4);  // -N per block
        cb_wait_front(cb_p, 4);
        tile_regs_acquire();
        copy_init(cb_c);
        for (uint32_t b = 0; b < 4; ++b) {
            copy_tile(cb_c, EYE, b);
        }
        put(cb_r, 4);  // R = I
        for (uint32_t step = 0; step < 31; ++step) {
            cb_wait_front(cb_r, 4);
            tile_regs_acquire();
            copy_init(cb_c);
            for (uint32_t b = 0; b < 4; ++b) {
                copy_tile(cb_c, EYE, b);
            }
            matmul_init(cb_p, cb_r);
            for (uint32_t b = 0; b < 4; ++b) {
                matmul_tiles(cb_p, cb_r, b, b, b);  // += (-N_b) @ R_b
            }
            put(step == 30 ? cb_out : cb_r, 4);
            cb_pop_front(cb_r, 4);
        }
        cb_pop_front(cb_p, 4);
        cb_pop_front(cb_sl, 4);
        cb_pop_front(cb_gcol, 4);
        cb_pop_front(cb_sfull, 1);
        cb_pop_front(cb_dcol, 4);
        cb_pop_front(cb_dexp, 4);
        cb_pop_front(cb_kdf, 4);
        cb_pop_front(cb_lm, 10);
        cb_pop_front(cb_kt, 16);
        cb_pop_front(cb_dinv, 4);
        cb_pop_front(cb_w1, 4);
        cb_pop_front(cb_w2, 4);
        cb_pop_front(cb_g, 4);
        cb_pop_front(cb_b, 4);
        cb_pop_front(cb_k, 16);
        cb_pop_front(cb_q, 16);
    }
}
