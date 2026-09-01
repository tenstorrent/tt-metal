// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // misc runtime args setup
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_src_bank = get_arg_val<uint32_t>(2);
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(3);
    std::uint32_t dram_buffer_dst_bank = get_arg_val<uint32_t>(4);
    std::uint32_t dram_buffer_size = get_arg_val<uint32_t>(5);
    std::uint64_t dram_buffer_src_noc_addr =
        get_noc_addr_from_bank_id<true>(dram_buffer_src_bank, dram_buffer_src_addr);

    // Below are some examples of dataflow_api.h function calls that are captured by the noc event profiler

    // this call will get captured by noc tracing as a 'READ' event
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
    // this call is captured as a 'READ_BARRIER_START' and 'READ_BARRIER_END' event ONLY when the NOC debug tool is
    // enabled (TT_METAL_NOC_DEBUG_DUMP). Barriers move no data, so plain noc tracing omits them -- NPE does not
    // consume them, and recording them costs runtime and profiler buffer space.
    noc_async_read_barrier();

    std::uint64_t dram_buffer_dst_noc_addr =
        get_noc_addr_from_bank_id<true>(dram_buffer_dst_bank, dram_buffer_dst_addr);

    // this call will get captured by noc tracing as a 'WRITE_' event
    noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
    // as with the read barrier above, this is captured as 'WRITE_BARRIER_START' / 'WRITE_BARRIER_END' only under
    // TT_METAL_NOC_DEBUG_DUMP
    noc_async_write_barrier();
}
