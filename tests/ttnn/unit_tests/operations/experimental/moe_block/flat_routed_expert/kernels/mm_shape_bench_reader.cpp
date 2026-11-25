// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Microbench reader: in0's n0 tiles into CB c_0 and in1's n1 tiles into c_1, once (the compute reuses them).
// Compile-time args: n0, n1, then the TensorAccessorArgs of in0 and in1. Runtime args: in0 address, in1 address.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t n0 = get_compile_time_arg_val(0);
    constexpr uint32_t n1 = get_compile_time_arg_val(1);
    constexpr auto a0 = TensorAccessorArgs<2>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const uint32_t p0 = get_local_cb_interface(0).fifo_page_size, p1 = get_local_cb_interface(1).fifo_page_size;
    const auto t0 = TensorAccessor(a0, get_arg_val<uint32_t>(0), p0);
    const auto t1 = TensorAccessor(a1, get_arg_val<uint32_t>(1), p1);
    cb_reserve_back(0, n0);
    cb_reserve_back(1, n1);
    for (uint32_t i = 0; i < n0; ++i) {
        noc_async_read(t0.get_noc_addr(i), get_write_ptr(0) + i * p0, p0);
    }
    for (uint32_t i = 0; i < n1; ++i) {
        noc_async_read(t1.get_noc_addr(i), get_write_ptr(1) + i * p1, p1);
    }
    noc_async_read_barrier();
    cb_push_back(0, n0);
    cb_push_back(1, n1);
}
