// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill concat-heads, writer: unit u = (user, h, r) -> tiles to out [1, 1, U * C, (H + 1) * 512] at tile row (user *
// ROWT + r), tile columns 16 * (1 + h) + j (the first 16 columns stay zero).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t ROWT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    constexpr uint32_t OCOLT = (H + 1) * 16;
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB c(cb);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t r = u % ROWT, h = (u / ROWT) % H, user = u / (ROWT * H);
        c.wait_front(16);
        for (uint32_t j = 0; j < 16; ++j) {
            noc.async_write(
                c,
                acc,
                2048,
                {.offset_bytes = j * 2048},
                {.page_id = (user * ROWT + r) * OCOLT + 16 * (1 + h) + j, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        c.pop_front(16);
    }
}
