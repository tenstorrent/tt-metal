// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill mHC "pre" — stream kernel (BRISC, NoC0): tensor reads and output writes (mhc_pre_pf_common.hpp).
//
// Member: fn^T + norm weight tiles of its columns once; per row block the X_TILES x tiles (flat page
//   rb * 512 + s * 128 + c0 + j -> CB index s * CPC + j, pushed one stream at a time so the compute's matmul starts on
//   the first stream), up to XBUF row blocks ahead of the hn writes; the CPC hn tiles (page rb * 128 + c0 + j) out.
// Root: the scale / bias row tiles and the identity tile once.
// Compile-time (member): CPC, XBUF, then the TensorAccessorArgs of x, fn, w, hn. Root: TensorAccessorArgs of sb.
// Runtime (member): x, fn, w, hn addresses, c0, first row block, row block stride, row blocks. Root: sb address.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_pre_pf_common.hpp"

using namespace mhcpf;

void kernel_main() {
#if defined(PF_MEMBER)
    constexpr uint32_t CPC = get_compile_time_arg_val(0);
    constexpr uint32_t XBUF = get_compile_time_arg_val(1);
    constexpr uint32_t X_TILES = N_STREAMS * CPC;
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto fn_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr auto hn_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto fn = TensorAccessor(fn_args, get_arg_val<uint32_t>(1));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(2));
    const auto hn = TensorAccessor(hn_args, get_arg_val<uint32_t>(3));
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const uint32_t rb0 = get_arg_val<uint32_t>(5);
    const uint32_t step = get_arg_val<uint32_t>(6);
    const uint32_t n_rb = get_arg_val<uint32_t>(7);

    // resident: fn^T rows of this core's K tiles (page = global K tile s * 128 + c) and the norm weight tiles
    cb_reserve_back(CB_FN, X_TILES);
    cb_reserve_back(CB_W, CPC);
    {
        const uint32_t l1 = get_write_ptr(CB_FN);
        const uint32_t fn_page = get_tile_size(CB_FN);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_read_page(s * COL_TILES + c0 + j, fn, l1 + (s * CPC + j) * fn_page);
            }
        }
        const uint32_t wl1 = get_write_ptr(CB_W);
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_read_page(c0 + j, w, wl1 + j * FP32_TILE_BYTES);
        }
    }
    noc_async_read_barrier();
    cb_push_back(CB_FN, X_TILES);
    cb_push_back(CB_W, CPC);

    auto read_block = [&](uint32_t rb) {
        cb_reserve_back(CB_X, X_TILES);
        cb_reserve_back(CB_X32, X_TILES);
        const uint32_t l1 = get_write_ptr(CB_X);
        const uint32_t base = rb * ROW_TILES + c0;
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_read_page(base + s * COL_TILES + j, x, l1 + (s * CPC + j) * FP32_TILE_BYTES);
            }
            noc_async_read_barrier();
            cb_push_back(CB_X, CPC);
            cb_push_back(CB_X32, CPC);
        }
    };

    uint32_t issued = rb0;  // next row block to read
    for (uint32_t rb = rb0; rb < n_rb; rb += step) {
        // keep up to XBUF row blocks in flight ahead of this block's hn (the reserve waits for the compute's pops)
        while (issued < n_rb && issued < rb + XBUF * step) {
            read_block(issued);
            issued += step;
        }
        cb_wait_front(CB_OUT, CPC);
        const uint32_t l1 = get_read_ptr(CB_OUT);
        const uint32_t page_bytes = get_tile_size(CB_OUT);
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_write_page(rb * COL_TILES + c0 + j, hn, l1 + j * page_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, CPC);
    }
#elif defined(PF_ROOT)
    constexpr auto sb_args = TensorAccessorArgs<0>();
    const auto sb = TensorAccessor(sb_args, get_arg_val<uint32_t>(0));
    cb_reserve_back(CB_SB, 3);
    const uint32_t l1 = get_write_ptr(CB_SB);
    for (uint32_t i = 0; i < 3; ++i) {
        noc_async_read_page(i, sb, l1 + i * FP32_TILE_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(CB_SB, 3);
#endif
}
