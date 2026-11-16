// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K2a head-core compute (see k2a_common.hpp; HiFi4, DEST fp32 as the engine's leaders): per layer, q head h's 32 rows
// (one row per launch row) -> q = rope(rmsnorm(q) w_q) into hcb_qo; the K / V role (sliding: h even k = rope(rmsnorm(k)
// w_k), h odd v = rmsnorm(v); full: h = 0 k = rope(rmsnorm(kv) w_k), h = 1 v = rmsnorm(kv)) as BF8 into hcb_kv8. The
// arithmetic is the engine's leader prep (attn_compute.cpp prep) for one head; every op is per row.
//
// Runtime args: 0 h (this head core's q head)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "k2a_common.hpp"

using namespace k2a;

namespace {

constexpr uint32_t kDst = 8;

// hcb_rinv = rsqrt(mean of the squares of rows over Dt tiles of cb_in from off + eps)
void rinv_rows(uint32_t cb_in, uint32_t off, uint32_t Dt, uint32_t inv_d) {
    cb_reserve_back(hcb_sq, 16);
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(hcb_sq);
    mul_init(cb_in, cb_in);
    for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < kDst && d0 + d < Dt; d++) {
            mul_tiles(cb_in, cb_in, off + d0 + d, off + d0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < kDst && d0 + d < Dt; d++) {
            pack_tile<true>(d, hcb_sq, d0 + d);
        }
        tile_regs_release();
    }
    cb_push_back(hcb_sq, 16);
    cb_wait_front(hcb_sq, 16);
    cb_reserve_back(hcb_rinv, 1);
    reconfig_data_format(hcb_one, hcb_sq);  // REDUCE_ROW SUM: scaler -> SrcA, data -> SrcB
    pack_reconfig_data_format(hcb_rinv);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(hcb_sq, hcb_one, hcb_rinv);
    tile_regs_acquire();
    for (uint32_t d = 0; d < Dt; d++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(hcb_sq, hcb_one, d, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_d);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, hcb_rinv);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(hcb_rinv, 1);
    cb_pop_front(hcb_sq, 16);
}

// cb_o (16 pages, tiles [0, Dt)) = cb_in[off + d] * rinv (row broadcast) [* hcb_w[w_off + d]]
void norm_apply(uint32_t cb_in, uint32_t off, uint32_t Dt, bool weighted, uint32_t w_off, uint32_t cb_o) {
    cb_wait_front(hcb_rinv, 1);
    cb_reserve_back(cb_o, 16);
    for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
        const uint32_t W = Dt - d0 < kDst ? Dt - d0 : kDst;
        tile_regs_acquire();
        reconfig_data_format(cb_in, hcb_rinv);
        mul_bcast_cols_init(cb_in, hcb_rinv);
        for (uint32_t d = 0; d < W; d++) {
            mul_tiles_bcast_cols(cb_in, hcb_rinv, off + d0 + d, 0, d);
        }
        if (weighted) {
            reconfig_data_format(hcb_w, hcb_w);
            mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(hcb_w);
            for (uint32_t d = 0; d < W; d++) {
                mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(hcb_w, w_off + d0 + d, d);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_o);
        for (uint32_t d = 0; d < W; d++) {
            pack_tile<true>(d, cb_o, d0 + d);
        }
        tile_regs_release();
    }
    cb_push_back(cb_o, 16);
    cb_pop_front(hcb_rinv, 1);
}

// cb_o[j] = x[j] cos[j] - x[j + H] sin[j] (j < H), x[j] cos[j] + x[j - H] sin[j] (j >= H), H = Dt / 2; x = cb_x
// tiles [0, Dt), cos / sin = hcb_cs tiles from c0 / s0 (row r: launch row r's position)
void rope(uint32_t cb_x, uint32_t Dt, uint32_t c0, uint32_t s0, uint32_t cb_o) {
    const uint32_t H = Dt / 2;
    cb_wait_front(cb_x, 16);
    for (uint32_t j0 = 0; j0 < Dt; j0 += kDst / 2) {
        tile_regs_acquire();
        reconfig_data_format(cb_x, hcb_cs);
        mul_init(cb_x, hcb_cs);
        for (uint32_t b = 0; b < kDst / 2; b++) {
            const uint32_t j = j0 + b, partner = j < H ? j + H : j - H;
            mul_tiles(cb_x, hcb_cs, j, c0 + j, 2 * b);
            mul_tiles(cb_x, hcb_cs, partner, s0 + j, 2 * b + 1);
        }
        if (j0 < H) {
            sub_binary_tile_init();
            for (uint32_t b = 0; b < kDst / 2; b++) {
                sub_binary_tile(2 * b, 2 * b + 1, 2 * b);
            }
        } else {
            add_binary_tile_init();
            for (uint32_t b = 0; b < kDst / 2; b++) {
                add_binary_tile(2 * b, 2 * b + 1, 2 * b);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_o);
        for (uint32_t b = 0; b < kDst / 2; b++) {
            pack_tile<true>(2 * b, cb_o, j0 + b);
        }
        tile_regs_release();
    }
}

// cb_o (16 pages) = tiles [0, n) of cb_in in cb_o's format
void convert(uint32_t cb_in, uint32_t n, uint32_t cb_o) {
    cb_wait_front(cb_in, 16);
    cb_reserve_back(cb_o, 16);
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_o);
    copy_init(cb_in);
    for (uint32_t d0 = 0; d0 < n; d0 += kDst) {
        tile_regs_acquire();
        for (uint32_t d = 0; d < kDst && d0 + d < n; d++) {
            copy_tile(cb_in, d0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < kDst && d0 + d < n; d++) {
            pack_tile<true>(d, cb_o, d0 + d);
        }
        tile_regs_release();
    }
    cb_push_back(cb_o, 16);
}

}  // namespace

void kernel_main() {
    const uint32_t h = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(hcb_rx, hcb_one, hcb_sq);
    cb_wait_front(hcb_one, 1);
    cb_wait_front(hcb_cs, kCSPages);
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t Dt = full ? kDtf : kDts;
        const uint32_t c0 = full ? 2 * kDts : 0, s0 = full ? 2 * kDts + kDtf : kDts;
        const uint32_t inv_d = full ? inv_df_bits : inv_ds_bits;
        // 0: Q only; 1: + K; 2: + V
        const uint32_t role = full ? (h == 0 ? 1 : (h == 1 ? 2 : 0)) : (h % 2 == 0 ? 1 : 2);
        cb_wait_front(hcb_w, 32);
        cb_wait_front(hcb_rx, kRx);
        // Q
        rinv_rows(hcb_rx, kRxQ, Dt, inv_d);
        norm_apply(hcb_rx, kRxQ, Dt, true, 0, hcb_xn);
        cb_reserve_back(hcb_qo, 16);
        rope(hcb_xn, Dt, c0, s0, hcb_qo);
        cb_push_back(hcb_qo, 16);
        cb_pop_front(hcb_xn, 16);
        // K / V
        if (role == 1 && !full) {
            rinv_rows(hcb_rx, kRxKV, Dt, inv_d);
            norm_apply(hcb_rx, kRxKV, Dt, true, 16, hcb_xn);
        } else if (role == 1) {
            rinv_rows(hcb_rx, kRxKV, Dt, inv_d);
            norm_apply(hcb_rx, kRxKV, Dt, false, 0, hcb_vn);
            // k = v * w_k (rounded to bf16 as the reference's multiply)
            cb_wait_front(hcb_vn, 16);
            cb_reserve_back(hcb_xn, 16);
            for (uint32_t d0 = 0; d0 < Dt; d0 += kDst) {
                tile_regs_acquire();
                reconfig_data_format(hcb_vn, hcb_w);
                mul_init(hcb_vn, hcb_w);
                for (uint32_t d = 0; d < kDst; d++) {
                    mul_tiles(hcb_vn, hcb_w, d0 + d, 16 + d0 + d, d);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(hcb_xn);
                for (uint32_t d = 0; d < kDst; d++) {
                    pack_tile<true>(d, hcb_xn, d0 + d);
                }
                tile_regs_release();
            }
            cb_push_back(hcb_xn, 16);
            cb_pop_front(hcb_vn, 16);
        } else if (role == 2) {
            rinv_rows(hcb_rx, kRxKV, Dt, inv_d);
            norm_apply(hcb_rx, kRxKV, Dt, false, 0, hcb_vn);
        }
        if (role == 1) {
            cb_reserve_back(hcb_kr, 16);
            rope(hcb_xn, Dt, c0, s0, hcb_kr);
            cb_push_back(hcb_kr, 16);
            cb_pop_front(hcb_xn, 16);
            convert(hcb_kr, Dt, hcb_kv8);
            cb_pop_front(hcb_kr, 16);
        } else if (role == 2) {
            convert(hcb_vn, Dt, hcb_kv8);
            cb_pop_front(hcb_vn, 16);
        }
        cb_pop_front(hcb_rx, kRx);
        cb_pop_front(hcb_w, 32);
    }
}
