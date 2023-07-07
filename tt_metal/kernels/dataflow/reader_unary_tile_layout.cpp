#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr             = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id    = get_arg_val<uint32_t>(1);
    uint32_t in0_num_blocks              = get_arg_val<uint32_t>(2);
    uint32_t in0_block_num_tiles         = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t tile_size_bytes = get_tile_size(cb_id_in0);
    // const args for tile-based bank-swizzled layout
    // could be added to the arg list in the future to test different
    // bank-swizzling configurations
    constexpr uint32_t num_used_dram_ch = 8;
    constexpr uint32_t num_used_dram_ch_pow2_exponent = 3;
    // TODO - pass this as an arg to the kernel
    constexpr uint32_t tile_size_pow2_exponent = 11;

    uint32_t tile_id = in0_tensor_start_tile_id;

    for (uint32_t block = 0; block < in0_num_blocks ; block++) {

        dataflow::cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        uint32_t l1_write_addr = dataflow::get_write_ptr(cb_id_in0);

        for(uint32_t tile = 0; tile < in0_block_num_tiles; tile++) {
            uint64_t in0_noc_addr = dataflow::get_noc_addr_for_tile(tile_id, in0_tensor_addr,
                                            num_used_dram_ch, num_used_dram_ch_pow2_exponent, tile_size_pow2_exponent);
            dataflow::noc_async_read(in0_noc_addr, l1_write_addr, tile_size_bytes);
            l1_write_addr += tile_size_bytes;
            tile_id++;
        }

        dataflow::noc_async_read_barrier();

        dataflow::cb_push_back(cb_id_in0, in0_block_num_tiles);
    }
}
