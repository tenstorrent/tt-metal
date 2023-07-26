#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr_base  = get_arg_val<uint32_t>(0);
    uint32_t src_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);
    uint32_t block_size_tiles = get_arg_val<uint32_t>(4);
    uint32_t block_size_bytes = get_arg_val<uint32_t>(5);
    uint32_t address_map_l1_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;
    volatile tt_l1_ptr std::uint32_t* address_map = (volatile tt_l1_ptr uint32_t*)(address_map_l1_addr);

    // Put zeroes in the zero buffer
    constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* zero_base_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }
    uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    uint32_t index = 0;
    // Read from DRAM into L1 using DTX address map and push one block at a time to CB
    for (uint32_t b = 0; b < num_blocks; b += 1) {
        cb_reserve_back(cb_id_in0, block_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint32_t bytes_read = 0;
        while(bytes_read != block_size_bytes) {
            // There are 3 entries in the address map per read
            uint32_t src_address = address_map[index];
            uint32_t dst_address = address_map[index+1];
            uint32_t read_size = address_map[index+2];
            uint32_t pad = address_map[index+3];
            if(pad) {
                // source address is set to max. This refers to padding location.
                // read zeroes from zero buffer
                uint64_t src_noc_addr = zeros_base_noc_addr;
                uint32_t dst_addr = l1_write_addr + dst_address;
                uint32_t pad_size = read_size;
                if (pad_size <= MEM_ZEROS_SIZE) {
                    noc_async_read(src_noc_addr, dst_addr, pad_size);
                }
                else {
                    // padding size is bigger than the zero buffer size
                    // read from zero buffer multiple times
                    uint32_t zeros_to_read = pad_size;
                    uint32_t zeros_read_size = MEM_ZEROS_SIZE;
                    while(zeros_to_read != 0) {
                        noc_async_read(src_noc_addr, dst_addr, zeros_read_size);
                        zeros_to_read -= zeros_read_size;
                        if (zeros_to_read < zeros_read_size) {
                            zeros_read_size = zeros_to_read;
                        }
                    }
                }
            }
            else {
                uint32_t src_addr = src_addr_base + src_address;
                uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);
                uint32_t dst_addr = l1_write_addr + dst_address;
                noc_async_read(src_noc_addr, dst_addr, read_size);
            }
            bytes_read += read_size;
            index += 4;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, block_size_tiles);
    }
}
