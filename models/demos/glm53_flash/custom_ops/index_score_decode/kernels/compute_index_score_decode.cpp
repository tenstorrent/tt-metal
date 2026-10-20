// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// index_score_decode compute (custom_ops/index_score_decode/op.py). Per chunk of KC k-tiles of one user:
//   S_c = relu(Q @ K_c^T)         one matmul_block (rt 1 x ct KC output tiles, kt = Dt inner tiles, K transposed),
//                                  relu in the packer -> cb_s [KC tiles of 32 heads x 32 pools]
//   out_c = Wrow @ S_c             one single-tile matmul per k-tile with the w-row tile (row 0 = the head weights,
//                                  rows 1.. zero), so row 0 of out_c = sum_h w_h relu(q_h . k_p) over the 32 pools
// The writer keeps row 0 of every out tile. bf16 DEST half-sync (KC <= 8 tiles in flight), as the shared scorer.
//
// compile-time args: [Dt, KC]
// runtime args: [item0, n_items, stride, n_bands, band_chunks, total_chunks]

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_w = 2, cb_s = 3, cb_out = 4;

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t n_bands = get_arg_val<uint32_t>(3);
    const uint32_t band_chunks = get_arg_val<uint32_t>(4);
    const uint32_t total_chunks = get_arg_val<uint32_t>(5);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);  // matmul: in1 -> srcA, in0 -> srcB

    for (uint32_t it = 0; it < n_items; ++it) {
        const uint32_t item = item0 + it * stride;
        const uint32_t band = item % n_bands;
        const uint32_t c_begin = band * band_chunks;
        const uint32_t c_end = (c_begin + band_chunks < total_chunks) ? (c_begin + band_chunks) : total_chunks;
        if (c_begin >= c_end) {
            continue;
        }
        cb_wait_front(cb_q, Dt);
        cb_wait_front(cb_w, 1);
        for (uint32_t c0 = c_begin; c0 < c_end; ++c0) {
            cb_wait_front(cb_k, KC * Dt);
            // phase 1: S = relu(Q K^T) for the chunk's KC k-tiles: one block (rt 1 x ct KC output tiles) per inner
            // d-tile, accumulated in DEST over the Dt inner tiles (matmul_block multiplies ONE inner tile per call;
            // the CB holds the chunk d-major: d-tile i of k-tile c at slot i*KC + c)
            // unpack formats: srcA <- k (bf16 or bfp8_b), srcB <- q (phase 2 left srcA on cb_s, srcB on cb_w)
            reconfig_data_format(cb_s, cb_k, cb_w, cb_q);
            matmul_block_init(cb_q, cb_k, 1 /*transpose k*/, KC /*ct*/, 1 /*rt*/, Dt /*kt*/);
            pack_relu_config(ReluConfig::zero());
            tile_regs_acquire();
            for (uint32_t i = 0; i < Dt; ++i) {
                matmul_block(cb_q, cb_k, i, i * KC, 0, 1 /*transpose*/, KC, 1, Dt);
            }
            tile_regs_commit();
            cb_reserve_back(cb_s, KC);
            tile_regs_wait();
            for (uint32_t j = 0; j < KC; ++j) {
                pack_tile(j, cb_s, j);
            }
            tile_regs_release();
            cb_push_back(cb_s, KC);
            cb_pop_front(cb_k, KC * Dt);
            // phase 2: out_j = Wrow @ S_j (row 0 = the head-weighted scores of k-tile j)
            pack_relu_config(ReluConfig::none());
            reconfig_data_format(cb_k, cb_s, cb_q, cb_w);  // srcA <- S (bf16), srcB <- the w-row tile
            matmul_block_init(cb_w, cb_s, 0, 1, 1, 1);
            cb_wait_front(cb_s, KC);
            tile_regs_acquire();
            for (uint32_t j = 0; j < KC; ++j) {
                matmul_block(cb_w, cb_s, 0, j, j, 0, 1, 1, 1);
            }
            tile_regs_commit();
            cb_pop_front(cb_s, KC);
            cb_reserve_back(cb_out, KC);
            tile_regs_wait();
            for (uint32_t j = 0; j < KC; ++j) {
                pack_tile(j, cb_out, j);
            }
            tile_regs_release();
            cb_push_back(cb_out, KC);
        }
        cb_pop_front(cb_q, Dt);
        cb_pop_front(cb_w, 1);
    }
}
