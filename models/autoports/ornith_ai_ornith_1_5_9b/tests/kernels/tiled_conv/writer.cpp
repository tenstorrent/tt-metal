// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    constexpr auto ha=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const uint32_t tile=get_arg_val<uint32_t>(0);
    const auto out=TensorAccessor(oa,get_arg_val<uint32_t>(1),2048);
    const auto hist=TensorAccessor(ha,get_arg_val<uint32_t>(2),2048);
    CircularBuffer result(16);result.wait_front(1);
    noc_async_write(get_read_ptr(16),out.get_noc_addr(tile),2048);
    // Compute retains these input pages until the writer completes. Each core
    // owns one channel tile in all three taps; no neighboring row is touched.
    noc_async_write(get_read_ptr(2),hist.get_noc_addr(tile),2048);
    noc_async_write(get_read_ptr(3),hist.get_noc_addr(64+tile),2048);
    noc_async_write(get_read_ptr(0),hist.get_noc_addr(128+tile),2048);
    noc_async_write_barrier();result.pop_front(1);
}
