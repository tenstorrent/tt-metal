// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unpatchify straight from the TILE-layout fp32 token tensor (1, S_pad, D), reader.
//
// Tokens of one (t, h) unit are the W = 16 consecutive rows (t*H + h)*16 .. +16: one half (two face rows) of a
// tile row. Inside a fp32 tile the 16 features of a 16-aligned group for those 16 tokens are one 1 KB face,
// which is exactly one output canvas row (c, frame t*pt + f, pixel row h*p + yy). So the unit's data is the
// 96 half-tiles of that tile row, staged here as 96 contiguous 2 KB reads; the writer only remaps pages.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(1);  // D / 32
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<7>();
    constexpr uint32_t TILE = 4096;
    constexpr uint32_t HALF = 2048;

    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    const auto in_acc = TensorAccessor(in_args, in_addr, TILE);
    experimental::CB stage(cb);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t s0 = u * 16;  // (t*H + h) * W with W = 16
        const uint32_t tile_row = s0 / 32;
        const uint32_t half = (s0 % 32) / 16;
        stage.reserve_back(1);
        for (uint32_t col = 0; col < d_tiles; ++col) {
            noc.async_read(
                in_acc,
                stage,
                HALF,
                {.page_id = tile_row * d_tiles + col, .offset_bytes = half * HALF},
                {.offset_bytes = col * HALF});
        }
        noc.async_read_barrier();
        stage.push_back(1);
    }
    (void)H;
}
