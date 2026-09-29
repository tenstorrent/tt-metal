// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the fused V4.1 mHC stream mix (mhc_mix_compute.cpp):  out_j = sum_k coef[j][k] (.) in_k, with one
// fp32 coefficient per token (tile row) and term.
//
// A work unit is one (tile row r, stream tile column c). Its K input tiles are, in order: the optional extra
// input x at (r, c) (HAS_X), then stream i at (r, i * CT + c) for i < N. Whenever r changes the reader reads
// tile row r of the (up to two) coefficient tensors and builds J * K column-broadcast tiles (tile j * K + k holds
// the coefficient of term k of output j in every column of its row), consumed by the compute kernel until the
// next row.
//
// compile_time_args = [cb_in, cb_coef, cb_csrc, CT, N, HAS_X, J, NUM_CSRC, BLOCK, table[J * K] (src << 8 | col),
//                      TensorAccessorArgs(streams), (x), (coef src 0), (coef src 1)...]
// runtime args      = [streams_addr, x_addr, csrc0_addr, csrc1_addr, unit_start, unit_count]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t streams_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t csrc0_addr = get_arg_val<uint32_t>(2);
    const uint32_t csrc1_addr = get_arg_val<uint32_t>(3);
    const uint32_t unit_start = get_arg_val<uint32_t>(4);
    const uint32_t unit_count = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(1);
    constexpr uint32_t cb_csrc = get_compile_time_arg_val(2);
    constexpr uint32_t CT = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_X = get_compile_time_arg_val(5);
    constexpr uint32_t J = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_CSRC = get_compile_time_arg_val(7);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(8);
    constexpr uint32_t K = N + HAS_X;
    constexpr uint32_t TABLE = 9;
    constexpr auto streams_args = TensorAccessorArgs<TABLE + J * K>();
    constexpr auto x_args = TensorAccessorArgs<streams_args.next_compile_time_args_offset()>();
    constexpr auto c0_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto c1_args = TensorAccessorArgs<c0_args.next_compile_time_args_offset()>();

    if (unit_count == 0) {
        return;
    }

    const auto streams = TensorAccessor(streams_args, streams_addr);
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto c0 = TensorAccessor(c0_args, csrc0_addr);
    const auto c1 = TensorAccessor(c1_args, csrc1_addr);

    Noc noc;
    DataflowBuffer in(cb_in);
    DataflowBuffer coef(cb_coef);
    DataflowBuffer csrc(cb_csrc);
    const uint32_t page = get_local_cb_interface(cb_in).fifo_page_size;  // fp32 tile
    constexpr uint32_t TILE_U32 = 32 * 32;
    constexpr uint32_t FACE_U32 = 16 * 16;

    csrc.reserve_back(NUM_CSRC);  // scratch, never pushed
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(csrc.get_write_ptr());

    const uint32_t unit_end = unit_start + unit_count;
    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t u = unit_start; u < unit_end;) {
        const uint32_t r = u / CT;
        const uint32_t c = u - r * CT;
        // a block: BLOCK units of one tile row (BLOCK divides CT and the host aligns every core's range to it), read
        // under one barrier; whole blocks keep every CB reservation contiguous
        constexpr uint32_t units = BLOCK;
        if (r != row) {
            row = r;
            noc.async_read(c0, csrc, page, {.page_id = r}, {.offset_bytes = 0});
            if constexpr (NUM_CSRC > 1) {
                noc.async_read(c1, csrc, page, {.page_id = r}, {.offset_bytes = page});
            }
            noc.async_read_barrier();
            coef.reserve_back(J * K);
            // plain (non-volatile) stores: the tiles are only read after push_back, so the compiler may schedule them
            tt_l1_ptr uint32_t* dst = reinterpret_cast<tt_l1_ptr uint32_t*>(coef.get_write_ptr());
            for (uint32_t t = 0; t < J * K; ++t) {
                const uint32_t code = kernel_compile_time_args[TABLE + t];
                const uint32_t col = code & 0xFFu;
                volatile tt_l1_ptr uint32_t* s = src + (code >> 8) * TILE_U32 + (col >> 4) * FACE_U32 + (col & 15u);
                tt_l1_ptr uint32_t* d = dst + t * TILE_U32;
                for (uint32_t tr = 0; tr < 32; ++tr) {
                    const uint32_t face = (tr >> 4) * 2;
                    const uint32_t off = (tr & 15u) * 16;
                    const uint32_t v = s[face * FACE_U32 + off];
                    tt_l1_ptr uint32_t* d0 = d + face * FACE_U32 + off;
                    tt_l1_ptr uint32_t* d1 = d0 + FACE_U32;
#pragma GCC unroll 16
                    for (uint32_t k = 0; k < 16; ++k) {
                        d0[k] = v;
                        d1[k] = v;
                    }
                }
            }
            asm volatile("" ::: "memory");  // the plain stores above complete before the tiles are published
            coef.push_back(J * K);
        }
        in.reserve_back(units * K);
        for (uint32_t b = 0; b < units; ++b) {
            const uint32_t base = b * K * page;
            if constexpr (HAS_X) {
                noc.async_read(xs, in, page, {.page_id = r * CT + c + b}, {.offset_bytes = base});
            }
            for (uint32_t i = 0; i < N; ++i) {
                noc.async_read(
                    streams,
                    in,
                    page,
                    {.page_id = r * (N * CT) + i * CT + c + b},
                    {.offset_bytes = base + (HAS_X + i) * page});
            }
        }
        noc.async_read_barrier();
        in.push_back(units * K);
        u += units;
    }
}
