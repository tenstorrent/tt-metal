// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Post-SDPA glue, rope cores: the rotated tile (row h = head h) -> row t of the output tiles (h, 14 + j) of
// [1,1,32,NH*512]: each 64 B row piece goes to the two faces of row t as 32 B writes.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t w = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t = w >> 1, j = w & 1;
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB out(cb_out);
    out.wait_front(1);
    const uint32_t dst = ((t >> 4) << 10) + ((t & 15) << 5);
    for (uint32_t h = 0; h < NH; ++h) {
        const uint32_t src = ((h >> 4) << 10) + ((h & 15) << 5);
        const uint32_t pg = h * 16 + 14 + j;
        noc.async_write(out, o_acc, 32, {.offset_bytes = src}, {.page_id = pg, .offset_bytes = dst});
        noc.async_write(out, o_acc, 32, {.offset_bytes = src + 512}, {.page_id = pg, .offset_bytes = dst + 512});
    }
    noc.async_write_barrier();
    out.pop_front(1);
}
