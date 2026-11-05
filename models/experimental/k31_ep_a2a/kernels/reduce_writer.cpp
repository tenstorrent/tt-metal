// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Combine reduction, writer: reduced 2 KB chunk -> out[t] at column chunk j.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr uint32_t CHUNK = 2048;
    constexpr uint32_t NCH = ROW_BYTES / CHUNK;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t u1 = get_arg_val<uint32_t>(2);
    const auto out = TensorAccessor(out_args, out_addr, ROW_BYTES);
    for (uint32_t u = u0; u < u1; ++u) {
        cb_wait_front(out_cb, 1);
        noc_async_write(get_read_ptr(out_cb), out.get_noc_addr(u / NCH, (u % NCH) * CHUNK), CHUNK);
        noc_async_writes_flushed();
        cb_pop_front(out_cb, 1);
    }
    noc_async_write_barrier();
}
