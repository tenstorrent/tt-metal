// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill q / o layout moves, reader (cores-for-prefill, tt/pf_retile.py). A unit is (head h, tile row r) of one user.
//   MODE 0 (tile -> row-major, q): reads the 16 tiles of x [U, H, C, 512] tile row (u, h, r) -> cb_in (16 tile pages)
//   MODE 1 (row-major -> tile, o): reads the 32 sticks (1 KB each) of o [1, 32, C, 512] rows (h, 32 r .. 32 r + 31) ->
//   cb_in (32 pages)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t MODE = get_compile_time_arg_val(1);
    constexpr uint32_t ROWT = get_compile_time_arg_val(2);  // tile rows per head (C / 32)
    constexpr uint32_t H = get_compile_time_arg_val(3);     // heads of the tile tensor (8)
    constexpr uint32_t C = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    const uint32_t user = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    experimental::CB cin(cb_in);
    if constexpr (MODE == 0) {
        const auto acc = TensorAccessor(x_args, x_addr, 2048);
        for (uint32_t u = u0; u < u0 + nu; ++u) {
            const uint32_t h = u / ROWT, r = u % ROWT;
            cin.reserve_back(16);
            for (uint32_t c = 0; c < 16; ++c) {
                noc.async_read(
                    acc,
                    cin,
                    2048,
                    {.page_id = ((user * H + h) * ROWT + r) * 16 + c, .offset_bytes = 0},
                    {.offset_bytes = c * 2048});
            }
            noc.async_read_barrier();
            cin.push_back(16);
        }
    } else {
        const auto acc = TensorAccessor(x_args, x_addr, 1024);
        for (uint32_t u = u0; u < u0 + nu; ++u) {
            const uint32_t h = u / ROWT, r = u % ROWT;
            cin.reserve_back(32);
            for (uint32_t i = 0; i < 32; ++i) {
                noc.async_read(
                    acc, cin, 1024, {.page_id = h * C + r * 32 + i, .offset_bytes = 0}, {.offset_bytes = i * 1024});
            }
            noc.async_read_barrier();
            cin.push_back(32);
        }
    }
}
