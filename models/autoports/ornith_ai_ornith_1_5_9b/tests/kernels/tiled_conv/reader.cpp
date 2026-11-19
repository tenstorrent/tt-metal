// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ra=TensorAccessorArgs<0>();
    constexpr auto ha=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto t0a=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto t1a=TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    constexpr auto t2a=TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    constexpr auto t3a=TensorAccessorArgs<t2a.next_compile_time_args_offset()>();
    const uint32_t tile=get_arg_val<uint32_t>(0);
    const auto raw=TensorAccessor(ra,get_arg_val<uint32_t>(1),2048);
    const auto hist=TensorAccessor(ha,get_arg_val<uint32_t>(2),2048);
    const auto t0=TensorAccessor(t0a,get_arg_val<uint32_t>(3),4096);
    const auto t1=TensorAccessor(t1a,get_arg_val<uint32_t>(4),4096);
    const auto t2=TensorAccessor(t2a,get_arg_val<uint32_t>(5),4096);
    const auto t3=TensorAccessor(t3a,get_arg_val<uint32_t>(6),4096);
    for(uint32_t cb=0;cb<8;++cb) { CircularBuffer buf(cb);buf.reserve_back(1); }
    noc_async_read(raw.get_noc_addr(tile),get_write_ptr(0),2048);
    for(uint32_t j=0;j<3;++j)noc_async_read(hist.get_noc_addr(j*64+tile),get_write_ptr(1+j),2048);
    noc_async_read(t0.get_noc_addr(tile),get_write_ptr(4),4096);
    noc_async_read(t1.get_noc_addr(tile),get_write_ptr(5),4096);
    noc_async_read(t2.get_noc_addr(tile),get_write_ptr(6),4096);
    noc_async_read(t3.get_noc_addr(tile),get_write_ptr(7),4096);
    noc_async_read_barrier();
    for(uint32_t cb=0;cb<8;++cb) { CircularBuffer buf(cb);buf.push_back(1); }
}
