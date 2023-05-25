#include <stdint.h>
#include "dataflow_api.h"
//#include "debug_print.h"

inline void async_read_from_dram_using_address_map(uint32_t dram_start_addr,
                                            uint32_t dram_noc_x,
                                            uint32_t dram_noc_y,
                                            uint32_t l1_write_addr,
                                            volatile uint32_t * address_map,
                                            uint32_t address_map_group_size,
                                            uint32_t address_map_group_start_index,
                                            uint32_t zeros_base_noc_addr) {
    for(uint32_t i = address_map_group_start_index; i < address_map_group_start_index + address_map_group_size; i+=4) {
        // There are 4 entries in the address map per read
        uint32_t src_address_offset = address_map[i];
        uint32_t dst_address_offset = address_map[i+1];
        uint32_t read_size = address_map[i+2];
        uint32_t pad = address_map[i+3];
        // DPRINT << "src_address_offset=" << src_address_offset << ENDL();
        // DPRINT << "dst_address_offset=" << dst_address_offset << ENDL();
        // DPRINT << "read_size=" << read_size << ENDL();
        // DPRINT << "pad=" << pad << ENDL();

        if(pad == 1) {
            // source address is set to max. This refers to padding location.
            // read zeroes from zero buffer
            uint32_t dst_addr = l1_write_addr + dst_address_offset;
            uint32_t pad_size = read_size;
            if (pad_size <= MEM_ZEROS_SIZE) {
                noc_async_read(zeros_base_noc_addr, dst_addr, pad_size);
            }
            else {
                // padding size is bigger than the zero buffer size
                // read from zero buffer multiple times
                uint32_t zeros_to_read = pad_size;
                uint32_t zeros_read_size = MEM_ZEROS_SIZE;
                while(zeros_to_read != 0) {
                    noc_async_read(zeros_base_noc_addr, dst_addr, zeros_read_size);
                    zeros_to_read -= zeros_read_size;
                    if (zeros_to_read < zeros_read_size) {
                        zeros_read_size = zeros_to_read;
                    }
                }
            }
        }
        else {
            uint32_t src_addr = dram_start_addr + src_address_offset;
            uint64_t src_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, src_addr);
            uint32_t dst_addr = l1_write_addr + dst_address_offset;
            noc_async_read(src_noc_addr, dst_addr, read_size);
        }
    }
}
void kernel_main() {
    // Arguments for in0
    uint32_t in0_addr_base  = get_arg_val<uint32_t>(0);
    uint32_t in0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t in0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t in0_address_map_l1_addr = get_arg_val<uint32_t>(3);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(4);

    // Arguments for in1
    uint32_t in1_addr_base  = get_arg_val<uint32_t>(5);
    uint32_t in1_noc_x = get_arg_val<uint32_t>(6);
    uint32_t in1_noc_y = get_arg_val<uint32_t>(7);
    uint32_t in1_address_map_l1_addr = get_arg_val<uint32_t>(8);
    uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    volatile std::uint32_t* in0_address_map = (volatile uint32_t*)(in0_address_map_l1_addr);
    volatile std::uint32_t* in1_address_map = (volatile uint32_t*)(in1_address_map_l1_addr);

    // Put zeroes in the zero buffer
    constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    volatile uint32_t* zero_base_ptr = reinterpret_cast<volatile uint32_t*>(MEM_ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }
    uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint32_t in0_address_map_index = 0;
    // address map vector contains number of groups in the first element
    uint32_t num_groups = in0_address_map[in0_address_map_index];
    in0_address_map_index += 1;
    // in0 and in1 address maps should have same number of groups
    uint32_t in1_address_map_index = 1;
    for(uint32_t g = 0; g < num_groups; g++) {
        //DPRINT << "group=" << g << ENDL();
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t in0_address_map_this_group_size = in0_address_map[in0_address_map_index];
        in0_address_map_index += 1;
        //DPRINT << "in0_address_map_index=" << in0_address_map_index << ENDL();

        async_read_from_dram_using_address_map(in0_addr_base, in0_noc_x, in0_noc_y,
            l1_write_addr_in0, in0_address_map, in0_address_map_this_group_size,
            in0_address_map_index, zeros_base_noc_addr);
        //DPRINT << "in0_address_map_this_group_size=" << in0_address_map_this_group_size << ENDL();
        noc_async_read_barrier();
        in0_address_map_index += in0_address_map_this_group_size;
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        uint32_t in1_address_map_this_group_size = in1_address_map[in1_address_map_index];
        in1_address_map_index += 1;
        //DPRINT << "in1_address_map_index=" << in1_address_map_index << ENDL();

        async_read_from_dram_using_address_map(in1_addr_base, in1_noc_x, in1_noc_y,
            l1_write_addr_in1, in1_address_map, in1_address_map_this_group_size,
            in1_address_map_index, zeros_base_noc_addr);
        in1_address_map_index += in1_address_map_this_group_size;
        //DPRINT << "in1_address_map_this_group_size=" << in1_address_map_this_group_size << ENDL();
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, in0_block_num_tiles);
        cb_push_back(cb_id_in1, in1_block_num_tiles);
    }
}
