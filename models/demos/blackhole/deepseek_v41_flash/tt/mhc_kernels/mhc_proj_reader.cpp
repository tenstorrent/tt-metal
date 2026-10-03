// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC projection, reader.  Core k owns the K-chunk (stream i = k / CPS, chunk c8 = k % CPS) of the flattened (stream,
// hidden) axis. It gathers, for each of the TPC 32-column tiles j of the chunk, a tile A_j with row t = x[t, i, 32
// cols] (rows 4..31 zero) and the matching weight tiles W_j (fn chunk, fp32). Also builds the "ones in column NCOL"
// tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// row r of a 32x32 tile: byte offset of its first face half (fp32) and word index
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))
#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);   // 32-col tiles per stream row (D/32)
    constexpr uint32_t TPC = get_compile_time_arg_val(5);  // tiles per chunk
    constexpr uint32_t CPS = get_compile_time_arg_val(6);  // chunks per stream
    constexpr uint32_t NCOL = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t i = k / CPS;
    const uint32_t c8 = k % CPS;

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cba(cb_a), cbw(cb_w), cbo(cb_ones);

    // ones-in-column-NCOL tile (all 32 rows): column c lives in face (c>=16) and face row r in face (r>=16)
    cbo.reserve_back(1);
    noc.async_write_zeros(cbo, TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t face = (r >= 16 ? 2 : 0) + (NCOL >= 16 ? 1 : 0);
            o[face * 256 + (r % 16) * 16 + (NCOL % 16)] = 0x3F800000u;
        }
    }
    cbo.push_back(1);

    // weight tiles: pages k*TPC .. k*TPC+TPC-1
    cbw.reserve_back(TPC);
    for (uint32_t j = 0; j < TPC; ++j) {
        noc.async_read(w_acc, cbw, TILE, {.page_id = k * TPC + j, .offset_bytes = 0}, {.offset_bytes = j * TILE});
    }

    // A tiles
    cba.reserve_back(TPC);
    noc.async_write_zeros(cba, TPC * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    for (uint32_t j = 0; j < TPC; ++j) {
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t page = t * NT + c8 * TPC + j;
            noc.async_read(
                x_acc, cba, 64, {.page_id = page, .offset_bytes = i * 64}, {.offset_bytes = j * TILE + RO(t)});
            noc.async_read(
                x_acc,
                cba,
                64,
                {.page_id = page, .offset_bytes = 1024 + i * 64},
                {.offset_bytes = j * TILE + 1024 + RO(t)});
        }
    }
    noc.async_read_barrier();
    cba.push_back(TPC);
    cbw.push_back(TPC);
}
