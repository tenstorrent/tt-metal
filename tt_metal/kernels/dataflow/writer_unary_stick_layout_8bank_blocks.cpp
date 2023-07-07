#include <stdint.h>
#include "dataflow_kernel_api.h"
#include "debug_print.h"

void kernel_main() {

    uint32_t dst_addr = get_arg_val<uint32_t>(0);           // out_dram_addr
    uint32_t num_rows_block = get_arg_val<uint32_t>(1);
    uint32_t block_row_size = get_arg_val<uint32_t>(2);     // in0_block_w * TILE_WIDTH * dtype_nbytes
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t num_blocks_h = get_arg_val<uint32_t>(4);
    uint32_t num_blocks_w = get_arg_val<uint32_t>(5);
    uint32_t output_row_size = get_arg_val<uint32_t>(6);    // in1_width * dtype_nbytes

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;

    constexpr uint32_t TILE_HEIGHT = 32;                    // TODO: use common source of truth
    constexpr uint32_t dtype_nbytes = 2;                    // TODO: obtain from data type

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t block_ntiles_w = block_row_size / 64; // Assuming 2 bytes per datum, there are 64 bytes per tile row
    const uint32_t block_ntiles_h = num_rows_block / TILE_HEIGHT;
    uint32_t start_block_row_id = 0;

    const dataflow::InterleavedAddrGen<true> s = {
        .bank_base_address = dst_addr,
        .page_size = output_row_size
    };
    for(uint32_t b = 0; b < batch; ++b) {
        for(uint32_t block_h = 0; block_h < num_blocks_h; block_h++) {
            uint32_t block_row_offset = 0;
            for(uint32_t block_w = 0; block_w < num_blocks_w; block_w++) {
                uint32_t block_row_id = start_block_row_id;
                for (uint32_t tile_row_id = 0; tile_row_id < block_ntiles_h; tile_row_id++) {
                    // We reserve back an entire row of tiles in a block and issue a bunch of reads
                    dataflow::cb_wait_front(cb_id_out0, block_ntiles_w);
                    uint32_t l1_read_addr = dataflow::get_read_ptr(cb_id_out0);
                    for (uint32_t j = 0; j < TILE_HEIGHT; j++) {
                        uint64_t dst_noc_addr = dataflow::get_noc_addr(block_row_id, s, block_row_offset);
                        dataflow::noc_async_write(l1_read_addr, dst_noc_addr, block_row_size);
                        l1_read_addr += block_row_size;
                        block_row_id++;
                    } // for tile_nrows
                    dataflow::noc_async_write_barrier();
                    dataflow::cb_pop_front(cb_id_out0, block_ntiles_w);
                } // for block_ntiles_h
                block_row_offset += block_row_size;
            } // for num_blocks_w
            start_block_row_id += num_rows_block;
        } // for num_blocks_h
    } // for batch
}
