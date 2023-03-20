#include <cstdint>
//#include "debug_print.h"

void kernel_main() {
    std::uint32_t dram_buffer_src_addr_base  = get_arg_val<uint32_t>(0);
    std::uint32_t dram_src_noc_x             = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_y             = get_arg_val<uint32_t>(2);
    std::uint32_t l1_buffer_dst_addr_base    = get_arg_val<uint32_t>(3);
    std::uint32_t address_map_l1_addr        = get_arg_val<uint32_t>(4);
    std::uint32_t address_map_size           = get_arg_val<uint32_t>(5);
    std::uint32_t num_tiles_c                = get_arg_val<uint32_t>(6);
    std::uint32_t tiles_c_bytes              = get_arg_val<uint32_t>(7);
    std::uint32_t total_bytes                = get_arg_val<uint32_t>(8);
    std::uint32_t row_size                   = get_arg_val<uint32_t>(9);
    std::uint32_t zero_buffer_l1_addr      = get_arg_val<uint32_t>(10);
    volatile std::uint32_t* address_map = (volatile uint32_t*)(address_map_l1_addr);
    volatile std::uint32_t* zero_buffer = (volatile uint32_t*)(zero_buffer_l1_addr);
    constexpr uint32_t cb_id_in0                       = 0;

    uint32_t i = 0;
    uint32_t total_bytes_written = 0;
    uint32_t bytes_written = 0;
    uint32_t l1_write_addr = l1_buffer_dst_addr_base;
    //DPRINT << "total bytes=" << total_bytes << " tiles_c_bytes=" << tiles_c_bytes << " num_tiles_c=" << num_tiles_c << " rows_size=" << row_size << ENDL();
    cb_reserve_back(cb_id_in0, num_tiles_c);
    while(total_bytes_written < total_bytes) {
        std::uint32_t dram_buffer_src_addr = dram_buffer_src_addr_base + address_map[i];
        std::uint32_t l1_buffer_dst_addr = l1_buffer_dst_addr_base + address_map[i+1];
        std::uint32_t dram_buffer_size = address_map[i+2];
        // DRAM NOC src address
        std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);
        noc_async_read(dram_buffer_src_noc_addr, l1_buffer_dst_addr, dram_buffer_size);
        l1_write_addr += dram_buffer_size;
        bytes_written += dram_buffer_size;
        total_bytes_written += dram_buffer_size;
        if(bytes_written == tiles_c_bytes) {
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, num_tiles_c);
            cb_reserve_back(cb_id_in0, num_tiles_c);
            bytes_written = 0;
        }
        i+=3;
    }
    noc_async_read_barrier();
    if(bytes_written != 0) {
        //H padding
        //DPRINT << "bytes_written=" << bytes_written << ENDL();
        while(bytes_written < tiles_c_bytes) {
            volatile std::uint32_t* dst = (volatile uint32_t*)(l1_write_addr);
            for(uint32_t z = 0; z < row_size / 4; z++) {
                dst[z] = zero_buffer[z];
            }
            l1_write_addr += row_size;
            bytes_written += row_size;
        }
        cb_push_back(cb_id_in0, num_tiles_c);
    }
}
