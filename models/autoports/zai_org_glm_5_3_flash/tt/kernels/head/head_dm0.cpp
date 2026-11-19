// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b model head, owner BRISC (NoC0): residual + norm weight in, h out (kernels/head/head_common.hpp).
// Compile-time args: TensorAccessorArgs of res, w, h. Runtime args: res (chunk 0), w, h addresses, c0 (first column
// tile), T, n readers, their NoC (x, y) (HEAD_HN_SEM incremented on each after h landed), then the residual addresses
// of chunks 1 .. (T - 1) / 4 (the same accessor layout as chunk 0).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/head_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"

using namespace head;

void kernel_main() {
    constexpr auto res_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const auto h = TensorAccessor(h_args, get_arg_val<uint32_t>(2));
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t T = get_arg_val<uint32_t>(4);
    const uint32_t n_readers = get_arg_val<uint32_t>(5);

    // flat x tiles: zero them (a DMA of the core's zero region), then 64-B face-row reads straight from the stacked
    // pages (flat (s, j) row t = stacked (t, c0 + j) row s, both face halves)
    cb_reserve_back(CB_X, N_STREAMS * COLS);
    const uint32_t x_l1 = get_write_ptr(CB_X);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < N_STREAMS * COLS * FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
#if defined(HEAD_RES_FLAT)
    // the FLAT residual [1, 1, T, 16384]: its (s, c0 + j) tiles are the flat x tiles already (rows = tokens)
    {
        const auto res = TensorAccessor(res_args, get_arg_val<uint32_t>(0));
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < COLS; ++j) {
                noc_async_read_page(s * COL_TILES + c0 + j, res, x_l1 + (s * COLS + j) * FP32_TILE_BYTES);
            }
        }
    }
#else
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t ch = t / CHUNK_ROWS;
        const auto res = TensorAccessor(res_args, get_arg_val<uint32_t>(ch == 0 ? 0 : 6 + 2 * n_readers + ch - 1));
        for (uint32_t j = 0; j < COLS; ++j) {
            const uint32_t page = (t % CHUNK_ROWS) * COL_TILES + c0 + j;
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                const uint32_t dst = x_l1 + (s * COLS + j) * FP32_TILE_BYTES;
                noc_async_read(res.get_noc_addr(page, tile_off(s, 0) * 4), dst + tile_off(t, 0) * 4, 64);
                noc_async_read(res.get_noc_addr(page, tile_off(s, 16) * 4), dst + tile_off(t, 16) * 4, 64);
            }
        }
    }
#endif
    cb_reserve_back(CB_W, COLS);
    const uint32_t w_l1 = get_write_ptr(CB_W);
    for (uint32_t j = 0; j < COLS; ++j) {
        noc_async_read_page(c0 + j, w, w_l1 + j * FP32_TILE_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(CB_X, N_STREAMS * COLS);
    cb_push_back(CB_W, COLS);

    // h out
    cb_wait_front(CB_OUT, COLS);
    const uint32_t o_l1 = get_read_ptr(CB_OUT);
    const uint32_t opage = get_tile_size(CB_OUT);
    for (uint32_t j = 0; j < COLS; ++j) {
        noc_async_write_page(c0 + j, h, o_l1 + j * opage);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, COLS);
    mk::signal_handoff(HEAD_HN_SEM, 6, n_readers);  // the lm_head readers' wait (count 64 owners)
}
