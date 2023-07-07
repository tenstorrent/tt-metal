#include <cstdint>
#include "dataflow_kernel_api.h"

void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t dram_dst_noc_x        = get_arg_val<uint32_t>(5);
    std::uint32_t dram_dst_noc_y        = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

    std::uint64_t dram_buffer_src_noc_addr = dataflow::get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);
    dataflow::noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
    dataflow::noc_async_read_barrier();

    std::uint64_t dram_buffer_dst_noc_addr = dataflow::get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);
    dataflow::noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
    dataflow::noc_async_write_barrier();
}
