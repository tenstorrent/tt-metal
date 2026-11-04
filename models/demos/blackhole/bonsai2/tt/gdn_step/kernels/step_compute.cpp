// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// One-token GatedDeltaNet step, compute: value head j (key head j / 3), K = V = 128 (4 tiles), fp32 DEST.
//
//   conv   c = silu(sum_r taps[r] * [state rows 0-2, x][r])      12 chunks of 32 channels (q, k, v of this head)
//   q, k   q = c_q / sqrt(|c_q|^2 + eps) * K^-1/2,  k = c_k / sqrt(|c_k|^2 + eps)
//   gates  decay = exp(A_neg * softplus(a + dt_bias)),  beta = sigmoid(b)
//   state  vread = decay * (k . S);  u = beta * (v - vread);  S' = decay * S + k^T (x) u;  o = q . S'
//   out    o / sqrt(mean(o^2) + eps) * w * silu(z)
//
// Every vector is a "row tile": row 0 holds the values, rows 1-31 are exactly zero (the reader zero-fills them and
// every step keeps them zero), so the rank-1 update k^T (x) u is a plain matmul and the readouts need no masking.
// Row sums use a matmul with an all-ones tile (every column of row 0 gets the sum); the conv's sum over rows 0-3
// uses a matmul with E (row 0 = [1, 1, 1, 1, 0, ...]).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"  // SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// silu on face rows 0-3 of faces 0 and 1 (SFPU rows 0, 1, 8, 9: row 0 is all a row tile holds) of DEST tiles 0..N-1:
// 4 SFPU rows per tile instead of 32. Needs silu_tile_init (Prgm0 = 2.0 for the reciprocal).
template <int N>
inline void silu_row0(uint32_t, uint32_t, uint32_t, uint32_t) {
    constexpr uint32_t OFF[4] = {0, 1, 8, 9};
#pragma GCC unroll 8
    for (int t = 0; t < N; ++t) {
#pragma GCC unroll 4
        for (int h = 0; h < 4; ++h) {
            sfpi::vFloat x = sfpi::dst_reg[t * 32 + OFF[h]];
            sfpi::dst_reg[t * 32 + OFF[h]] = x * _sfpu_sigmoid_<true>(x);
        }
    }
}
inline void silu_row0_4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { silu_row0<4>(a, b, c, d); }
inline void silu_row0_8(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { silu_row0<8>(a, b, c, d); }
}  // namespace ckernel::sfpu
#endif
// clang-format off
#define SILU_ROW0(FN) MATH((SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, FN, 0, 0, 0, 0, VectorMode::None)))
// clang-format on

#ifdef STEP_ZONES
#include "api/debug/kernel_profiler.hpp"
#define ZONE(n) DeviceZoneScopedN(n)
#else
#define ZONE(n)
#endif

namespace {
constexpr uint32_t cb_x = 0, cb_t = 1, cb_z = 2, cb_a = 3, cb_b = 4, cb_s = 5, cb_w = 6, cb_e = 7, cb_ones = 8;
constexpr uint32_t cb_p = 11, cb_conv = 12, cb_sq = 13, cb_inv = 14, cb_qk = 15, cb_out = 16, cb_decay = 17;
constexpr uint32_t cb_beta = 18, cb_kcol = 19, cb_u = 20, cb_snew = 21, cb_o = 22, cb_zs = 23, cb_oinv = 24;
constexpr uint32_t HT = 4, NCH = 12, DST = 8;
constexpr uint32_t ONE_F = 0x3F800000u, TWENTY_F = 0x41A00000u;

ALWI void fmt(uint32_t a, uint32_t b) { reconfig_data_format(a, b); }
ALWI void fmt_mm(uint32_t in0, uint32_t in1) { reconfig_data_format<SrcOrder::Reverse>(in0, in1); }

// Pack DST tiles 0..n-1 to cb positions base..base+n-1 (cb already reserved).
ALWI void pack_n(uint32_t n, uint32_t cb, uint32_t base) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile<true>(i, cb, base + i);
    }
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    const uint32_t dt_bias = get_arg_val<uint32_t>(0);  // fp32 bits
    const uint32_t a_neg = get_arg_val<uint32_t>(1);
    const uint32_t eps_qk = get_arg_val<uint32_t>(2);
    const uint32_t q_scale = get_arg_val<uint32_t>(3);
    const uint32_t inv_hd = get_arg_val<uint32_t>(4);
    const uint32_t eps_o = get_arg_val<uint32_t>(5);

    compute_kernel_hw_startup(cb_a, cb_b, cb_decay);

    {
        ZONE("TD-gates");
        // gates
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_decay, 1);
        cb_reserve_back(cb_beta, 1);
        tile_regs_acquire();
        fmt(cb_a, cb_a);
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        fmt(cb_b, cb_b);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 1);
        binop_with_scalar_tile_init();
        add_unary_tile(0, dt_bias);
        softplus_tile_init();
        softplus_tile(0, ONE_F, ONE_F, TWENTY_F);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, a_neg);
        exp_tile_init();
        exp_tile(0);
        sigmoid_tile_init();
        sigmoid_tile(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_decay);
        pack_tile<true>(0, cb_decay, 0);
        pack_tile<true>(1, cb_beta, 0);
        tile_regs_release();
        cb_push_back(cb_decay, 1);
        cb_push_back(cb_beta, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
    {
        ZONE("TD-siluz");
        // silu(z)
        cb_wait_front(cb_z, HT);
        cb_reserve_back(cb_zs, HT);
        tile_regs_acquire();
        fmt(cb_z, cb_z);
        copy_init(cb_z);
        for (uint32_t i = 0; i < HT; ++i) {
            copy_tile(cb_z, i, i);
        }
        silu_tile_init();
        SILU_ROW0(silu_row0_4);
        pack_n(HT, cb_zs, 0);
        cb_push_back(cb_zs, HT);
        cb_pop_front(cb_z, HT);
    }
    {
        ZONE("TD-conv");
        // conv: P_c = X_c * T_c, then c = silu(E @ P_c)
        cb_wait_front(cb_x, NCH);
        cb_wait_front(cb_t, NCH);
        cb_reserve_back(cb_p, NCH);
        for (uint32_t b = 0; b < NCH; b += DST) {
            const uint32_t n = NCH - b < DST ? NCH - b : DST;
            tile_regs_acquire();
            fmt(cb_x, cb_t);
            mul_init(cb_x, cb_t);
            for (uint32_t i = 0; i < n; ++i) {
                mul_tiles(cb_x, cb_t, b + i, b + i, i);
            }
            pack_n(n, cb_p, b);
        }
        cb_push_back(cb_p, NCH);
        cb_pop_front(cb_x, NCH);
        cb_pop_front(cb_t, NCH);
        cb_wait_front(cb_p, NCH);
        cb_wait_front(cb_e, 1);
        cb_reserve_back(cb_conv, NCH);
        for (uint32_t b = 0; b < NCH; b += DST) {
            const uint32_t n = NCH - b < DST ? NCH - b : DST;
            tile_regs_acquire();
            fmt_mm(cb_e, cb_p);
            matmul_init(cb_e, cb_p);
            for (uint32_t i = 0; i < n; ++i) {
                matmul_tiles(cb_e, cb_p, 0, b + i, i);
            }
            silu_tile_init();
            if (n == 8) {
                SILU_ROW0(silu_row0_8);
            } else {
                SILU_ROW0(silu_row0_4);
            }
            pack_n(n, cb_conv, b);
        }
        cb_push_back(cb_conv, NCH);
        cb_pop_front(cb_p, NCH);
        cb_pop_front(cb_e, 1);
    }
    {
        ZONE("TD-qknorm");
        // q, k L2 norms
        cb_wait_front(cb_conv, NCH);
        cb_reserve_back(cb_sq, 2 * HT);
        tile_regs_acquire();
        fmt(cb_conv, cb_conv);
        mul_init(cb_conv, cb_conv);
        for (uint32_t i = 0; i < 2 * HT; ++i) {
            mul_tiles(cb_conv, cb_conv, i, i, i);
        }
        pack_n(2 * HT, cb_sq, 0);
        cb_push_back(cb_sq, 2 * HT);
        cb_wait_front(cb_sq, 2 * HT);
        cb_wait_front(cb_ones, 1);
        cb_reserve_back(cb_inv, 2);
        tile_regs_acquire();
        fmt_mm(cb_sq, cb_ones);
        matmul_init(cb_sq, cb_ones);
        for (uint32_t i = 0; i < 2 * HT; ++i) {
            matmul_tiles(cb_sq, cb_ones, i, 0, i / HT);
        }
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_qk);
        add_unary_tile(1, eps_qk);
        rsqrt_tile_init();
        rsqrt_tile(0);
        rsqrt_tile(1);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, q_scale);
        pack_n(2, cb_inv, 0);
        cb_push_back(cb_inv, 2);
        cb_pop_front(cb_sq, 2 * HT);
        cb_wait_front(cb_inv, 2);
        cb_reserve_back(cb_qk, 2 * HT);
        tile_regs_acquire();
        fmt(cb_conv, cb_inv);
        mul_init(cb_conv, cb_inv);
        for (uint32_t i = 0; i < 2 * HT; ++i) {
            mul_tiles(cb_conv, cb_inv, i, i / HT, i);
        }
        pack_n(2 * HT, cb_qk, 0);
        cb_push_back(cb_qk, 2 * HT);
        cb_pop_front(cb_inv, 2);
    }
    {
        ZONE("TD-kcol");
        // k as columns (for the rank-1 update)
        cb_wait_front(cb_qk, 2 * HT);
        cb_reserve_back(cb_kcol, HT);
        tile_regs_acquire();
        fmt(cb_qk, cb_qk);
        transpose_init(cb_qk);
        for (uint32_t i = 0; i < HT; ++i) {
            transpose_tile(cb_qk, HT + i, i);
        }
        pack_n(HT, cb_kcol, 0);
        cb_push_back(cb_kcol, HT);
    }
    {
        ZONE("TD-u");
        // u = beta * (v - decay * (k . S))
        cb_wait_front(cb_s, HT * HT);
        cb_wait_front(cb_decay, 1);
        cb_wait_front(cb_beta, 1);
        cb_reserve_back(cb_u, HT);
        tile_regs_acquire();
        fmt_mm(cb_qk, cb_s);
        matmul_init(cb_qk, cb_s);
        for (uint32_t n = 0; n < HT; ++n) {
            for (uint32_t k = 0; k < HT; ++k) {
                matmul_tiles(cb_qk, cb_s, HT + k, k * HT + n, n);
            }
        }
        fmt(cb_decay, cb_decay);
        copy_init(cb_decay);
        copy_tile(cb_decay, 0, HT);
        sfpu_mul_bcast_col_init();
        for (uint32_t n = 0; n < HT; ++n) {
            sfpu_mul_bcast_col(n, HT);
        }
        fmt(cb_conv, cb_conv);
        sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_conv);
        for (uint32_t n = 0; n < HT; ++n) {
            sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_conv, 2 * HT + n, n);  // v - vread
        }
        fmt(cb_beta, cb_beta);
        copy_init(cb_beta);
        copy_tile(cb_beta, 0, HT);
        sfpu_mul_bcast_col_init();
        for (uint32_t n = 0; n < HT; ++n) {
            sfpu_mul_bcast_col(n, HT);
        }
        pack_n(HT, cb_u, 0);
        cb_push_back(cb_u, HT);
    }
    {
        ZONE("TD-snew");
        // S' = decay * S + k^T (x) u  (bf16 out: the writer stores it, the readout below uses it)
        cb_wait_front(cb_kcol, HT);
        cb_wait_front(cb_u, HT);
        cb_reserve_back(cb_snew, HT * HT);
        for (uint32_t b = 0; b < HT * HT; b += DST) {
            tile_regs_acquire();
            fmt(cb_s, cb_decay);
            mul_bcast_scalar_init(cb_s, cb_decay);
            for (uint32_t i = 0; i < DST; ++i) {
                mul_tiles_bcast_scalar(cb_s, cb_decay, b + i, 0, i);
            }
            fmt_mm(cb_kcol, cb_u);
            matmul_init(cb_kcol, cb_u);
            for (uint32_t i = 0; i < DST; ++i) {
                matmul_tiles(cb_kcol, cb_u, (b + i) / HT, (b + i) % HT, i);
            }
            pack_n(DST, cb_snew, b);
        }
        cb_push_back(cb_snew, HT * HT);
    }
    {
        ZONE("TD-o");
        // o = q . S'
        cb_wait_front(cb_snew, HT * HT);  // not popped: the writer reads it too
        cb_reserve_back(cb_o, HT);
        tile_regs_acquire();
        fmt_mm(cb_qk, cb_snew);
        matmul_init(cb_qk, cb_snew);
        for (uint32_t n = 0; n < HT; ++n) {
            for (uint32_t k = 0; k < HT; ++k) {
                matmul_tiles(cb_qk, cb_snew, k, k * HT + n, n);
            }
        }
        pack_n(HT, cb_o, 0);
        cb_push_back(cb_o, HT);
        cb_pop_front(cb_qk, 2 * HT);
        cb_pop_front(cb_kcol, HT);
        cb_pop_front(cb_u, HT);
        cb_pop_front(cb_s, HT * HT);
        cb_pop_front(cb_decay, 1);
        cb_pop_front(cb_beta, 1);
        cb_pop_front(cb_conv, NCH);
    }
    {
        ZONE("TD-out");
        // out = o / sqrt(mean(o^2) + eps) * w * silu(z)
        cb_wait_front(cb_o, HT);
        cb_reserve_back(cb_sq, HT);
        tile_regs_acquire();
        fmt(cb_o, cb_o);
        mul_init(cb_o, cb_o);
        for (uint32_t i = 0; i < HT; ++i) {
            mul_tiles(cb_o, cb_o, i, i, i);
        }
        pack_n(HT, cb_sq, 0);
        cb_push_back(cb_sq, HT);
        cb_wait_front(cb_sq, HT);
        cb_reserve_back(cb_oinv, 1);
        tile_regs_acquire();
        fmt_mm(cb_sq, cb_ones);
        matmul_init(cb_sq, cb_ones);
        for (uint32_t i = 0; i < HT; ++i) {
            matmul_tiles(cb_sq, cb_ones, i, 0, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_hd);
        add_unary_tile(0, eps_o);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack_n(1, cb_oinv, 0);
        cb_push_back(cb_oinv, 1);
        cb_pop_front(cb_sq, HT);
        cb_pop_front(cb_ones, 1);
        cb_wait_front(cb_oinv, 1);
        cb_wait_front(cb_w, HT);
        cb_wait_front(cb_zs, HT);
        cb_reserve_back(cb_out, HT);
        tile_regs_acquire();
        fmt(cb_o, cb_oinv);
        mul_init(cb_o, cb_oinv);
        for (uint32_t i = 0; i < HT; ++i) {
            mul_tiles(cb_o, cb_oinv, i, 0, i);
        }
        fmt(cb_w, cb_w);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
        for (uint32_t i = 0; i < HT; ++i) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, i, i);
        }
        fmt(cb_zs, cb_zs);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_zs);
        for (uint32_t i = 0; i < HT; ++i) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_zs, i, i);
        }
        pack_n(HT, cb_out, 0);
        cb_push_back(cb_out, HT);
        cb_pop_front(cb_o, HT);
        cb_pop_front(cb_oinv, 1);
        cb_pop_front(cb_w, HT);
        cb_pop_front(cb_zs, HT);
    }
}
