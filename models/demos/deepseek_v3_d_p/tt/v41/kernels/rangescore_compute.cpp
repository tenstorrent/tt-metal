// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-query range scoring of a DeepSeek-V4.1 index source (tt/v41/indexer_kernels.py range_scores): one user's
// query against the visible prefix of its own index keys, the key tile rows split over the cores in chunks.
//
// Once per core: Q = the user's [32 heads, 128] query (4 tiles [heads, dims], tilized from the reader's row-major
// block), W = a tile with its 32 head weights in row 0 (zeros below). Per key tile row of 32 keys (K tiles [rows,
// dims] straight from the tiled pool):
//   S = relu(Q @ K^T)          [heads, rows]: fp32 accumulation over the 128 dims, relu + bf16 rounding in the packer
//   O = W @ S                  row 0 = sum_h w[h] * S[h, :] (fp32 accumulation over the heads, HiFi4)
// which is candidate_scores's arithmetic (candscore_compute.cpp). The writer takes row 0 of every O tile. The
// visible tile-row count comes from the reader through cb_meta (word 0); chunk c belongs to core c mod NCORES, and
// a chunk at or past the bound is not computed. CB traffic is in whole chunks (CHUNK result tiles pushed, the
// first n meaningful) so the pointers never straddle the wrap.
//
// compile_time_args = [CHUNK, TR, NCORES]; runtime args = [core_index].

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
    constexpr uint32_t CHUNK = get_compile_time_arg_val(0);
    constexpr uint32_t TR = get_compile_time_arg_val(1);
    constexpr uint32_t NCORES = get_compile_time_arg_val(2);
    const uint32_t core = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_qrm = tt::CBIndex::c_0;
    constexpr uint32_t cb_wrm = tt::CBIndex::c_1;
    constexpr uint32_t cb_k = tt::CBIndex::c_2;
    constexpr uint32_t cb_meta = tt::CBIndex::c_3;
    constexpr uint32_t cb_q = tt::CBIndex::c_4;
    constexpr uint32_t cb_w = tt::CBIndex::c_5;
    constexpr uint32_t cb_s = tt::CBIndex::c_7;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t DIM_TILES = 4;
    constexpr uint32_t NCHUNKS = (TR + CHUNK - 1) / CHUNK;

    DataflowBuffer q(cb_q);
    DataflowBuffer w(cb_w);
    DataflowBuffer kt(cb_k);
    DataflowBuffer meta(cb_meta);
    DataflowBuffer s(cb_s);
    DataflowBuffer out(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_s);
    matmul_init(cb_q, cb_k, 1);
    MATH((llk_math_reconfig_remap(true)));

    meta.wait_front(1);
    const uint32_t vt = meta.read_tile_value(0, 0);
    meta.pop_front(1);

    tilize_blocks<DIM_TILES, cb_qrm, cb_q>(1);
    tilize_blocks<1, cb_wrm, cb_w>(1);
    q.wait_front(DIM_TILES);
    w.wait_front(1);

    for (uint32_t c = core; c < NCHUNKS; c += NCORES) {
        const uint32_t t0 = c * CHUNK;
        if (t0 >= vt) {
            break;
        }
        uint32_t t1 = t0 + CHUNK;
        if (t1 > vt) {
            t1 = vt;
        }
        const uint32_t n = t1 - t0;

        matmul_init(cb_q, cb_k, 1);
        pack_relu_config(ReluConfig::zero());
        kt.wait_front(DIM_TILES * CHUNK);
        s.reserve_back(CHUNK);
        for (uint32_t b = 0; b < n; ++b) {
            tile_regs_acquire();
            for (uint32_t d = 0; d < DIM_TILES; ++d) {
                matmul_tiles(cb_q, cb_k, d, b * DIM_TILES + d, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_s, b);
            tile_regs_release();
        }
        s.push_back(CHUNK);
        kt.pop_front(DIM_TILES * CHUNK);
        pack_relu_config(ReluConfig::none());

        matmul_init(cb_w, cb_s, 0);
        s.wait_front(CHUNK);
        out.reserve_back(CHUNK);
        for (uint32_t b = 0; b < n; ++b) {
            tile_regs_acquire();
            matmul_tiles(cb_w, cb_s, 0, b, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_out, b);
            tile_regs_release();
        }
        out.push_back(CHUNK);
        s.pop_front(CHUNK);
    }
    q.pop_front(DIM_TILES);
    w.pop_front(1);
}
