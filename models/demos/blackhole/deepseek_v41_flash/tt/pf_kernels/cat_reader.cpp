// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill concat-heads, reader: unit u = (user, head h, tile row r) of o [U, H, C, 512] bf16 TILE -> its 16 tiles into
// cb (pass-through to the writer).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t ROWT = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto acc = TensorAccessor(x_args, x_addr, 2048);
    experimental::CB c(cb);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        c.reserve_back(16);
        for (uint32_t j = 0; j < 16; ++j) {
            noc.async_read(acc, c, 2048, {.page_id = u * 16 + j, .offset_bytes = 0}, {.offset_bytes = j * 2048});
        }
        noc.async_read_barrier();
        c.push_back(16);
    }
}
