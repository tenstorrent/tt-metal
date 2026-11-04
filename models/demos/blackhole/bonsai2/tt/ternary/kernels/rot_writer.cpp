// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode rotation, writer (BRISC): the rotated block (row-major, 2048 B) into each of the C copies of x', at 2048 b.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t b = get_arg_val<uint32_t>(1);
    constexpr uint32_t C = get_compile_time_arg_val(0);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(out_args, out_addr, ROW_BYTES);
    constexpr uint32_t cb_y = tt::CBIndex::c_16;
    cb_wait_front(cb_y, 1);
    const uint32_t l1 = get_read_ptr(cb_y);
    for (uint32_t c = 0; c < C; ++c) {
        noc_async_write(l1, out.get_noc_addr(c, b * 2048), 2048);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_y, 1);
}
