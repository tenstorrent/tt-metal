// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void process_head(uint32_t h) {
    constexpr auto oa=TensorAccessorArgs<3>();
    constexpr auto sa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const uint32_t batch=h/8,hv=h%8;
    const auto out=TensorAccessor(oa,get_arg_val<uint32_t>(2),2048);
    const auto state=TensorAccessor(sa,get_arg_val<uint32_t>(3),4096);
    CircularBuffer result(31),final_state(7);result.wait_front(4);
    const uint32_t row_offset=(batch/16)*1024+(batch%16)*32;
    for(uint32_t t=0;t<4;++t) {
        const uint64_t target=out.get_noc_addr(hv*4+t);
        noc_async_write(get_read_ptr(31)+t*2048,target+row_offset,32);
        noc_async_write(get_read_ptr(31)+t*2048+512,target+row_offset+512,32);
    }
    noc_async_write_barrier();result.pop_front(4);
    final_state.wait_front(16);
    for(uint32_t t=0;t<16;++t)noc_async_write(get_read_ptr(7)+t*4096,state.get_noc_addr(h*16+t),4096);
    noc_async_write_barrier();final_state.pop_front(16);
}

void kernel_main() {
    const uint32_t first=get_arg_val<uint32_t>(0), heads=get_arg_val<uint32_t>(4), stride=get_arg_val<uint32_t>(5);
    for(uint32_t head=first;head<heads;head+=stride) process_head(head);
}
