// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Packed-layout mHC projection, reader.  Core k owns K tiles [k*TPC, (k+1)*TPC) of the 4*CT K-tiles of the packed
// stream row (stream i = tile/CT); it keeps their fn^T weight tiles in L1 and streams, for every token tile row r, its
// TPC x tiles.  Also builds the ones-in-column-NCOL tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t R = get_compile_time_arg_val(3);
    constexpr uint32_t XT = get_compile_time_arg_val(4);  // K tiles per token tile row (4*CT)
    constexpr uint32_t TPC = get_compile_time_arg_val(5);
    constexpr uint32_t NCOL = get_compile_time_arg_val(6);
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    const auto w_acc = TensorAccessor(w_args, w_addr, TILE);
    experimental::CB cbx(cb_x), cbw(cb_w), cbo(cb_ones);

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

    cbw.reserve_back(TPC);
    for (uint32_t j = 0; j < TPC; ++j) {
        noc.async_read(w_acc, cbw, TILE, {.page_id = k * TPC + j, .offset_bytes = 0}, {.offset_bytes = j * TILE});
    }
    noc.async_read_barrier();
    cbw.push_back(TPC);

    for (uint32_t r = 0; r < R; ++r) {
        cbx.reserve_back(TPC);
        for (uint32_t j = 0; j < TPC; ++j) {
            noc.async_read(
                x_acc, cbx, TILE, {.page_id = r * XT + k * TPC + j, .offset_bytes = 0}, {.offset_bytes = j * TILE});
        }
        noc.async_read_barrier();
        cbx.push_back(TPC);
    }
}
