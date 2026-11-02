// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill q / o layout moves, writer.
//   MODE 0: 32 sticks of the unit -> out [1, 32, C, 512] row-major pages (h * C + 32 r + i)
//   MODE 1: 16 tiles of the unit -> out [U, H, C, 512] tile pages (((user * H + h) * ROWT + r) * 16 + c)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t MODE = get_compile_time_arg_val(1);
    constexpr uint32_t ROWT = get_compile_time_arg_val(2);
    constexpr uint32_t H = get_compile_time_arg_val(3);
    constexpr uint32_t C = get_compile_time_arg_val(4);
    constexpr auto o_args = TensorAccessorArgs<5>();
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    const uint32_t user = get_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    experimental::CB out(cb_out);
    if constexpr (MODE == 0) {
        const auto acc = TensorAccessor(o_args, o_addr, 1024);
        for (uint32_t u = u0; u < u0 + nu; ++u) {
            const uint32_t h = u / ROWT, r = u % ROWT;
            out.wait_front(16);
            for (uint32_t i = 0; i < 32; ++i) {
                noc.async_write(
                    out, acc, 1024, {.offset_bytes = i * 1024}, {.page_id = h * C + r * 32 + i, .offset_bytes = 0});
            }
            noc.async_write_barrier();
            out.pop_front(16);
        }
    } else {
        const auto acc = TensorAccessor(o_args, o_addr, 2048);
        for (uint32_t u = u0; u < u0 + nu; ++u) {
            const uint32_t h = u / ROWT, r = u % ROWT;
            out.wait_front(16);
            for (uint32_t c = 0; c < 16; ++c) {
                noc.async_write(
                    out,
                    acc,
                    2048,
                    {.offset_bytes = c * 2048},
                    {.page_id = ((user * H + h) * ROWT + r) * 16 + c, .offset_bytes = 0});
            }
            noc.async_write_barrier();
            out.pop_front(16);
        }
    }
}
