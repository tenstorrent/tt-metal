#include <stdint.h>
#include "dataflow_api.h"
void kernel_main() {
    // Arguments for in1
    uint32_t src1_addr  = get_arg_val<uint32_t>(0);
    uint32_t in1_tiles  = get_arg_val<uint32_t>(1);
    // Arguments for in0
    uint32_t src0_addr  = get_arg_val<uint32_t>(2);
    uint32_t in0_tiles  = get_arg_val<uint32_t>(3);
    uint32_t num_bytes_of_zeroes_per_transfer = get_arg_val<uint32_t>(4);
    uint32_t num_transfers_of_zeroes = get_arg_val<uint32_t>(5);
    uint32_t address_map_l1_addr = get_arg_val<uint32_t>(6);
    uint32_t address_map_size = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    volatile tt_l1_ptr std::uint32_t* address_map = (volatile tt_l1_ptr uint32_t*)(address_map_l1_addr);
    constexpr uint32_t num_elements_in_zeros_buffer = MEM_ZEROS_SIZE / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* zero_base_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_ZEROS_BASE);
    for (uint32_t zero_base_offset = 0; zero_base_offset < num_elements_in_zeros_buffer; zero_base_offset++) {
        *(zero_base_ptr + zero_base_offset) = 0;
    }
    uint64_t zeros_base_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    constexpr uint32_t tile_size_pow2_exponent = 11;

    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = src0_addr,


        .page_size = address_map[2] // size of 1 stick = transfer size in address map = number of channels
    };

    const InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = src1_addr,


        .log_base_2_of_page_size = tile_size_pow2_exponent
    };

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    cb_reserve_back(cb_id_in0, in0_tiles);
    cb_reserve_back(cb_id_in1, in1_tiles);
    l1_write_addr_in1 = get_write_ptr(cb_id_in1);
    // Read weights
    for(uint32_t in1_tile = 0; in1_tile < in1_tiles; in1_tile+=1) {
        uint64_t in1_tile_noc_addr = get_noc_addr(in1_tile, s1);
        noc_async_read(in1_tile_noc_addr, l1_write_addr_in1, single_tile_size_bytes);
        l1_write_addr_in1 += single_tile_size_bytes;
    }

    // Read activations using address map
    l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t index = 0;
    for (uint32_t index = 0; index < address_map_size; index+=4) {
        uint32_t src_addr = src0_addr + address_map[index];
        // Destination address at address_map[index+1] unused. Contiguous writes to L1.
        uint32_t size = address_map[index+2];
        uint32_t row_bank_id = address_map[index+3];
        uint64_t in0_row_noc_addr = get_noc_addr(row_bank_id, s0);
        noc_async_read(in0_row_noc_addr, l1_write_addr_in0, size);
        l1_write_addr_in0 += size;
    }
    // Height padding
    for (uint32_t z = 0; z < num_transfers_of_zeroes; z++) {
        noc_async_read(zeros_base_noc_addr, l1_write_addr_in1, num_bytes_of_zeroes_per_transfer);
        l1_write_addr_in0 += num_bytes_of_zeroes_per_transfer;
    }

    noc_async_read_barrier();
    cb_push_back(cb_id_in0, in0_tiles);
    cb_push_back(cb_id_in1, in1_tiles);
}
