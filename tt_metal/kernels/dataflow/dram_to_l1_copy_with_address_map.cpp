#include <cstdint>

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or other RISCs
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t dram_buffer_src_addr_base  = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src_noc_x             = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_y             = get_arg_val<uint32_t>(2);
    std::uint32_t l1_buffer_dst_addr_base    = get_arg_val<uint32_t>(3);
    std::uint32_t address_map_l1_addr        = get_arg_val<uint32_t>(4);
    std::uint32_t address_map_size           = get_arg_val<uint32_t>(5);

    volatile tt_l1_ptr std::uint32_t* address_map = (volatile tt_l1_ptr uint32_t*)(address_map_l1_addr);


    for(std::uint32_t i = 0; i < address_map_size; i+=3) {
        std::uint32_t dram_buffer_src_addr = dram_buffer_src_addr_base + address_map[i];
        std::uint32_t l1_buffer_dst_addr = l1_buffer_dst_addr_base + address_map[i+1];
        std::uint32_t dram_buffer_size = address_map[i+2];
        // DRAM NOC src address
        std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);
        noc_async_read(dram_buffer_src_noc_addr, l1_buffer_dst_addr, dram_buffer_size);
        noc_async_read_barrier();

    }

}
