// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Post-SDPA glue, pass-through cores: units u = h*T + t (head h, user t): the first 448 dims of SDPA output row (h, t)
// go, as 14 x (2 x 32 B), to row t of the output tiles (h, 0..13) of [1,1,32,NH*512].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_p = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t ROW = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    constexpr auto d_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t cnt = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_common_arg_val<uint32_t>(1);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, ROW);
    const auto d_acc = TensorAccessor(d_args, d_addr, 2048);
    experimental::CB cbp(cb_p);
    cbp.reserve_back(1);
    for (uint32_t u = u0; u < u0 + cnt; ++u) {
        const uint32_t h = u / T, t = u - h * T;
        noc.async_read(o_acc, cbp, 896, {.page_id = u, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t dst = ((t >> 4) << 10) + ((t & 15) << 5);
        for (uint32_t k = 0; k < 14; ++k) {
            noc.async_write(cbp, d_acc, 32, {.offset_bytes = k * 64}, {.page_id = h * 16 + k, .offset_bytes = dst});
            noc.async_write(
                cbp, d_acc, 32, {.offset_bytes = k * 64 + 32}, {.page_id = h * 16 + k, .offset_bytes = dst + 512});
        }
        noc.async_write_barrier();
    }
}
