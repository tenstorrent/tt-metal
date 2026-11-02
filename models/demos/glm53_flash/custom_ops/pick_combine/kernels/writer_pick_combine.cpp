// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pick_combine writer (custom_ops/pick_combine/op.py): per job the sw packed output tiles of (block b, slice s) go
// to the [1, 1, T, H] bf16 TILE output, tile page b * H_tiles + s * sw + c.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace wct {
enum : uint32_t { CB_OUT = 0, SW, H_TILES, SLICES, END };
}

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(wct::CB_OUT);
    constexpr uint32_t sw = get_compile_time_arg_val(wct::SW);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(wct::H_TILES);
    constexpr uint32_t slices = get_compile_time_arg_val(wct::SLICES);
    constexpr auto out_args = TensorAccessorArgs<wct::END>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t job0 = get_arg_val<uint32_t>(1);
    const uint32_t jobs = get_arg_val<uint32_t>(2);
    constexpr uint32_t tile_bytes = 2048;
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    for (uint32_t job = job0; job < job0 + jobs; ++job) {
        const uint32_t b = job / slices;
        const uint32_t s = job % slices;
        cb_wait_front(cb_out, sw);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < sw; ++c) {
            noc_async_write(src + c * tile_bytes, out.get_noc_addr(b * h_tiles + s * sw + c), tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, sw);
    }
}
