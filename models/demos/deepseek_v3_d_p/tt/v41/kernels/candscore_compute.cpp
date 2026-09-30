// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Candidate scoring of a DeepSeek-V4.1 candidate index source (tt/v41/indexer_kernels.py candidate_scores).
//
// Per query: Q = its [32 heads, 128] query (4 tiles [heads, dims]), W = a tile with its 32 head weights in row 0
// (zeros below). Per tile-row c of 32 candidate rows (tilized [rows, dims] K tiles):
//   S = relu(Q @ K^T)          [heads, rows]: fp32 accumulation over the 128 dims, relu + bf16 rounding in the packer
//   O = W @ S                  row 0 = sum_h w[h] * S[h, :] (fp32 accumulation over the heads, HiFi4: exact bf16
//                              products), packed bf16
// which is the reference's score order (per head a bf16 q . k, relu, weighted sum over the heads). The writer takes
// row 0 of every O tile. Tile-rows are processed in batches of BATCH (one tilize / matmul / matmul phase each).
//
// compile_time_args = [K, BATCH]; runtime args = [row_count].

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace tcfg = compute_kernel_lib::tilize_config;

template <uint32_t WIDTH, uint32_t IN, uint32_t OUT>
ALWI void tilize_blocks(uint32_t blocks) {
    compute_kernel_lib::tilize<
        WIDTH,
        IN,
        OUT,
        tcfg::InitUninitMode::InitAndUninit,
        tcfg::WaitMode::WaitBlock,
        tcfg::ReconfigureRegisterDatatypeMode::NoReconfigure,
        tcfg::Fp32Mode::Fast,
        tcfg::RemapMode::AssumeConfigured>(blocks);
}

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);
    const uint32_t row_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_qrm = tt::CBIndex::c_0;
    constexpr uint32_t cb_wrm = tt::CBIndex::c_1;
    constexpr uint32_t cb_krm = tt::CBIndex::c_2;
    constexpr uint32_t cb_q = tt::CBIndex::c_4;
    constexpr uint32_t cb_w = tt::CBIndex::c_5;
    constexpr uint32_t cb_k = tt::CBIndex::c_6;
    constexpr uint32_t cb_s = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t DIM_TILES = 4;
    constexpr uint32_t TILE_ROWS = K * 8 / 32;
    static_assert(TILE_ROWS % BATCH == 0);

    DataflowBuffer q(cb_q);
    DataflowBuffer w(cb_w);
    DataflowBuffer kt(cb_k);
    DataflowBuffer s(cb_s);
    DataflowBuffer out(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
    matmul_init(cb_q, cb_k, 1);
    MATH((llk_math_reconfig_remap(true)));

    for (uint32_t row = 0; row < row_count; ++row) {
        tilize_blocks<DIM_TILES, cb_qrm, cb_q>(1);
        tilize_blocks<1, cb_wrm, cb_w>(1);
        q.wait_front(DIM_TILES);
        w.wait_front(1);
        for (uint32_t c0 = 0; c0 < TILE_ROWS; c0 += BATCH) {
            tilize_blocks<DIM_TILES, cb_krm, cb_k>(BATCH);

            matmul_init(cb_q, cb_k, 1);
            pack_relu_config(ReluConfig::zero());
            kt.wait_front(DIM_TILES * BATCH);
            s.reserve_back(BATCH);
            for (uint32_t b = 0; b < BATCH; ++b) {
                tile_regs_acquire();
                for (uint32_t d = 0; d < DIM_TILES; ++d) {
                    matmul_tiles(cb_q, cb_k, d, b * DIM_TILES + d, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_s, b);
                tile_regs_release();
            }
            s.push_back(BATCH);
            kt.pop_front(DIM_TILES * BATCH);
            pack_relu_config(ReluConfig::none());

            matmul_init(cb_w, cb_s, 0);
            s.wait_front(BATCH);
            out.reserve_back(BATCH);
            for (uint32_t b = 0; b < BATCH; ++b) {
                tile_regs_acquire();
                matmul_tiles(cb_w, cb_s, 0, b, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_out, b);
                tile_regs_release();
            }
            out.push_back(BATCH);
            s.pop_front(BATCH);
        }
        q.pop_front(DIM_TILES);
        w.pop_front(1);
    }
}
