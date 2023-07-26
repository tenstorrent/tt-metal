#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr_base  = get_arg_val<uint32_t>(0);
    uint32_t src_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t block_size_tiles = get_arg_val<uint32_t>(4); // block row of tiles
    uint32_t unpadded_last_row_tiles_H = get_arg_val<uint32_t>(5);
    uint32_t num_bytes_of_zeroes_per_transfer = get_arg_val<uint32_t>(6);
    uint32_t num_transfers_of_zeroes = get_arg_val<uint32_t>(7);
    uint32_t remainder_zeroes = get_arg_val<uint32_t>(8);
    uint32_t address_map_l1_addr = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = 0;
    volatile tt_l1_ptr std::uint32_t* address_map = (volatile tt_l1_ptr uint32_t*)(address_map_l1_addr);
    constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* zero_base_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }
    uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t block_size_bytes = get_tile_size(cb_id_in0) * block_size_tiles;
    uint32_t row_size_bytes = 64 * block_size_tiles;
    uint32_t index = 0;
    // read a block of tiles row-wise from src to CB, and then push the block to unpacker
    for (uint32_t t = 0; t < num_tiles - block_size_tiles; t += block_size_tiles) {
        cb_reserve_back(cb_id_in0, block_size_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t i = 0; i < 32; i++) {
            uint32_t bytes_written = 0;
            while(bytes_written < row_size_bytes) {
                std::uint32_t src_addr = src_addr_base + address_map[index];
                // Address map's destination address unused. Contiguous writes to L1.
                //std::uint32_t l1_dst_addr = l1_write_addr + address_map[index+1];
                std::uint32_t size = address_map[index+2];
                index += 3;
                uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);
                noc_async_read(src_noc_addr, l1_write_addr, size);
                l1_write_addr += size;
                bytes_written += size;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, block_size_tiles);
    }

    // Read last block of tiles and pad height
    cb_reserve_back(cb_id_in0, block_size_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    for(uint32_t i = 0; i < unpadded_last_row_tiles_H; i++) {
        uint32_t bytes_written = 0;
        while(bytes_written < row_size_bytes) {
            std::uint32_t src_addr = src_addr_base + address_map[index];
            // Address map's destination address unused. Contiguous writes to L1.
            //std::uint32_t l1_dst_addr = l1_write_addr + address_map[index+1];
            std::uint32_t size = address_map[index+2];
            index += 3;
            uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);
            noc_async_read(src_noc_addr, l1_write_addr, size);
            l1_write_addr += size;
            bytes_written += size;
        }
    }
    // H padding for last block of tiles
    for (uint32_t z = 0; z < num_transfers_of_zeroes; z++) {
        noc_async_read(zeros_base_noc_addr, l1_write_addr, num_bytes_of_zeroes_per_transfer);
        l1_write_addr += num_bytes_of_zeroes_per_transfer;
    }
    if(remainder_zeroes > 0) {
        noc_async_read(zeros_base_noc_addr, l1_write_addr, remainder_zeroes);
        l1_write_addr += remainder_zeroes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, block_size_tiles);
}
