// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pick_combine compute (custom_ops/pick_combine/op.py): per job (block b, slice s) and pick j the reader's [32, S]
// row-major block is tilized (sw tiles; all k picks first, tilize goes through DST) and multiplied by the block's
// diagonal weight tile diag(w[:, j]) into the sw fp32 DST tiles, accumulating over the k picks: out[t, :] = sum_j w[t,
// j] * row(pos[t*k + j])[:], packed to cb_out as bf16. The k weight tiles stay in cb_w for all the slices of a block
// (popped on its last job).
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"

namespace cct {
enum : uint32_t { CB_G_RM = 0, CB_G, CB_W, CB_OUT, K, SW, SLICES };
}

void kernel_main() {
    constexpr uint32_t cb_g_rm = get_compile_time_arg_val(cct::CB_G_RM);
    constexpr uint32_t cb_g = get_compile_time_arg_val(cct::CB_G);
    constexpr uint32_t cb_w = get_compile_time_arg_val(cct::CB_W);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cct::CB_OUT);
    constexpr uint32_t k = get_compile_time_arg_val(cct::K);
    constexpr uint32_t sw = get_compile_time_arg_val(cct::SW);
    constexpr uint32_t slices = get_compile_time_arg_val(cct::SLICES);
    const uint32_t job0 = get_arg_val<uint32_t>(0);
    const uint32_t jobs = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_w, cb_g, cb_out);
    uint32_t cur_block = 0xFFFFFFFFu;
    for (uint32_t job = job0; job < job0 + jobs; ++job) {
        const uint32_t b = job / slices;
        if (b != cur_block) {
            if (cur_block != 0xFFFFFFFFu) {
                cb_pop_front(cb_w, k);
            }
            cur_block = b;
            cb_wait_front(cb_w, k);
        }
        // the k pick blocks tilized first (tilize goes through DST), then the accumulating matmuls
        for (uint32_t j = 0; j < k; ++j) {
            cb_wait_front(cb_g_rm, sw);
            cb_reserve_back(cb_g, sw);
            reconfig_data_format_srca(cb_g_rm);
            pack_reconfig_data_format(cb_g);
            tilize_init(cb_g_rm, sw, cb_g);
            tilize_block(cb_g_rm, sw, cb_g);
            tilize_uninit(cb_g_rm, cb_g);
            cb_push_back(cb_g, sw);
            cb_pop_front(cb_g_rm, sw);
        }
        cb_wait_front(cb_g, k * sw);
        tile_regs_acquire();
        // the DST accumulators start from whatever the tilize left there: zero them (the expert stream's fill_tile)
        fill_tile_init();
        for (uint32_t c = 0; c < sw; ++c) {
            fill_tile(c, 0.0f);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_w, cb_g);
        matmul_init(cb_w, cb_g);
        for (uint32_t j = 0; j < k; ++j) {
            for (uint32_t c = 0; c < sw; ++c) {
                matmul_tiles(cb_w, cb_g, j, j * sw + c, c);
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_g, k * sw);
        cb_reserve_back(cb_out, sw);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        for (uint32_t c = 0; c < sw; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, sw);
    }
    if (cur_block != 0xFFFFFFFFu) {
        cb_pop_front(cb_w, k);
    }
}
