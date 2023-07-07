#include <cstdint>

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or other RISCs
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t l1_buffer_addr             = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr_base  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_x             = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src_noc_y             = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_dst_addr_base  = get_arg_val<uint32_t>(4);
    std::uint32_t dram_dst_noc_x             = get_arg_val<uint32_t>(5);
    std::uint32_t dram_dst_noc_y             = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size           = get_arg_val<uint32_t>(7);
    std::uint32_t chunk_size                 = get_arg_val<uint32_t>(8);

    // loading_noc variable is defined by either NCRISC or BRISC to be 0 or 1, depending on which RISC the kernel is running

    for (std::uint32_t offset = 0; offset < dram_buffer_size; offset += chunk_size) {
        std::uint32_t dram_buffer_src_addr = dram_buffer_src_addr_base + offset;
        // DRAM NOC src address
        std::uint64_t dram_buffer_src_noc_addr = dataflow::get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);
        dataflow::noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, chunk_size);
        dataflow::noc_async_read_barrier();

        // DRAM NOC dst address
        std::uint32_t dram_buffer_dst_addr = dram_buffer_dst_addr_base + offset;
        std::uint64_t dram_buffer_dst_noc_addr = dataflow::get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);
        dataflow::noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, chunk_size);
        dataflow::noc_async_write_barrier();
    }
}
