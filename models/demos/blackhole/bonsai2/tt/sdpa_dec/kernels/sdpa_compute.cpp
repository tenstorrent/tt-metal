// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode SDPA (sdpa_dec.py), compute. Core (g, c) with nr K / V tile rows: S_t = Q K_t^T (Dt tiles each, K transposed
// by the unpacker) -> cb_s (fp32; the writer masks the last); its row maxima (reduce MAX) -> cb_m; with the group's
// global maxima M (cb_M, from the writer): P_t = exp(scale (S_t - M)) -> cb_p; O_d = sum_t P_t V_{t,d} and
// l = sum_t P_t 1 (the row sums in every column) -> cb_o. Combiner d (c < NC): out = (Sel stack_O) / (Sel stack_l),
// Sel summing the 5 x 6-row partials of each stacked tile into rows 0 .. 5.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_binary_sfpu.h"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_one = 3, cb_info = 5, cb_tok = 7, cb_M = 8, cb_stk = 9;
constexpr uint32_t cb_s = 16, cb_m = 17, cb_o = 18, cb_out = 19, cb_p = 20;
constexpr uint32_t ONES = 0, SCALER = 1, SEL = 2, Dt = 8;

ALWI void put(uint32_t cb, uint32_t n, uint32_t d0 = 0) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    cb_reserve_back(cb, n);
    for (uint32_t d = 0; d < n; ++d) {
        pack_tile(d0 + d, cb);
    }
    cb_push_back(cb, n);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    const uint32_t c = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_common_arg_val<uint32_t>(0), NST = get_common_arg_val<uint32_t>(1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
    cb_wait_front(cb_info, 1);
    const uint32_t nr = read_tile_value(cb_info, 0, 0);
    if (nr != 0) {
        cb_wait_front(cb_q, Dt);
        cb_wait_front(cb_k, nr * Dt);
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_q, cb_k);
        matmul_init(cb_q, cb_k, 1);
        for (uint32_t t = 0; t < nr; ++t) {
            for (uint32_t d = 0; d < Dt; ++d) {
                matmul_tiles(cb_q, cb_k, d, t * Dt + d, t);
            }
        }
        put(cb_s, nr);
        cb_wait_front(cb_one, 3);  // (the writer's constant tiles, made while S ran)
        cb_wait_front(cb_tok, 1);  // masked
        cb_wait_front(cb_s, nr);
        tile_regs_acquire();
        reconfig_data_format(cb_s, cb_one);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, cb_m);
        for (uint32_t t = 0; t < nr; ++t) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_one, t, SCALER, 0);
        }
        reduce_uninit();
        put(cb_m, 1);
    }
    cb_wait_front(cb_M, 1);
    if (nr != 0) {
        tile_regs_acquire();
        reconfig_data_format(cb_s, cb_M);
        sub_bcast_cols_init(cb_s, cb_M);
        for (uint32_t t = 0; t < nr; ++t) {
            sub_tiles_bcast_cols(cb_s, cb_M, t, 0, t);
        }
        exp_tile_init<false>();
        for (uint32_t t = 0; t < nr; ++t) {
            // rows 0 .. 5 only: faces 0 / 1, the first EXP_ITERS SFPU row pairs
            exp_tile<false, true, InputClamping::ClampToNegative, EXP_ITERS>(t, VectorMode::R, SCALE_BF16);
        }
        put(cb_p, nr);
        cb_wait_front(cb_p, nr);
        cb_wait_front(cb_v, nr * Dt);
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_p, cb_v);
        matmul_init(cb_p, cb_v, 0);
        for (uint32_t d = 0; d < Dt; ++d) {
            for (uint32_t t = 0; t < nr; ++t) {
                matmul_tiles(cb_p, cb_v, t, t * Dt + d, d);
            }
        }
        put(cb_o, Dt);
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_p, cb_one);
        matmul_init(cb_p, cb_one, 0);
        for (uint32_t t = 0; t < nr; ++t) {
            matmul_tiles(cb_p, cb_one, t, ONES, 0);
        }
        put(cb_o, 1);
    }
    if (c < NC) {
        cb_wait_front(cb_one, 3);
        cb_wait_front(cb_stk, 2 * NST);
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_one, cb_stk);
        matmul_init(cb_one, cb_stk, 0);
        for (uint32_t t = 0; t < NST; ++t) {
            matmul_tiles(cb_one, cb_stk, SEL, t, 0);
        }
        for (uint32_t t = 0; t < NST; ++t) {
            matmul_tiles(cb_one, cb_stk, SEL, NST + t, 1);
        }
        recip_tile_init();
        recip_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        put(cb_out, 1);
    }
}
