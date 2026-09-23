// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unpatchify from the TILE-layout fp32 token tensor, writer: face r of the staged unit (r = (c*pt + f)*p + yy,
// 1 KB) is canvas row (c, t*pt + f, h*p + yy) of the (1, C, T*pt, H*p, W*p) row-major output, one 1 KB page.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t C = get_compile_time_arg_val(4);
    constexpr uint32_t PT = get_compile_time_arg_val(5);
    constexpr uint32_t P = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr uint32_t ROW = 1024;  // W * P * 4 bytes with W = P = 16
    constexpr uint32_t ROWS = C * PT * P;

    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    const auto out_acc = TensorAccessor(out_args, out_addr, ROW);
    experimental::CB stage(cb);
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t t = u / H;
        const uint32_t h = u % H;
        stage.wait_front(1);
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t c = r / (PT * P);
            const uint32_t f = (r / P) % PT;
            const uint32_t yy = r % P;
            const uint32_t page = (c * T * PT + t * PT + f) * (H * P) + h * P + yy;
            noc.async_write(stage, out_acc, ROW, {.offset_bytes = r * ROW}, {.page_id = page, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        stage.pop_front(1);
    }
}
