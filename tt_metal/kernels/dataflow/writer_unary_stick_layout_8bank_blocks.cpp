#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"
void kernel_main() {

    // Constexpr
    constexpr uint32_t num_dram_channels               = 8;
    constexpr uint32_t log_base_2_of_num_dram_channels = 3;
    constexpr uint32_t cb_id_out0                      = 16;

    uint32_t dst_addr                 = get_arg_val<uint32_t>(0);
    uint32_t num_rows_block               = get_arg_val<uint32_t>(1);
    uint32_t block_row_size               = get_arg_val<uint32_t>(2);
    uint32_t batch_size             = get_arg_val<uint32_t>(3);
    uint32_t num_blocks_h           = get_arg_val<uint32_t>(4);
    uint32_t num_blocks_w           = get_arg_val<uint32_t>(5);
    uint32_t output_row_size = get_arg_val<uint32_t>(6);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_block_c = block_row_size / 64; // Assuming 2 bytes per datum, there are 64 bytes per tile row
    uint32_t start_block_row_id          = 0;

    const InterleavedAddrGen<true> s = {
        .bank_base_address = dst_addr,


        .page_size = output_row_size
    };

    kernel_profiler::mark_time(5);

    for(uint32_t batch = 0; batch < batch_size; batch++) {
        for(uint32_t block_h = 0; block_h < num_blocks_h; block_h++) {
            uint32_t block_row_offset = 0;
            for(uint32_t block_w = 0; block_w < num_blocks_w; block_w++) {
    // kernel_profiler::mark_time(5);
                uint32_t block_row_id = start_block_row_id;
                for (uint32_t i = 0; i < num_rows_block / 32; i++) {
                    // We reserve back an entire tile row and issue a bunch of reads
                    cb_wait_front(cb_id_out0, num_tiles_block_c);
                // kernel_profiler::mark_time(5);

                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    for (uint32_t j = 0; j < 32; j++) {
                        uint64_t dst_noc_addr = get_noc_addr(
                            block_row_id, s, block_row_offset);

                        uint32_t bank_id = block_row_id & (num_dram_channels - 1);
                        noc_async_write(l1_read_addr, dst_noc_addr, block_row_size);
                        l1_read_addr += block_row_size;
                        block_row_id++;
                    }
                kernel_profiler::mark_time(6);
                    noc_async_write_barrier();
                // kernel_profiler::mark_time(6);
                    cb_pop_front(cb_id_out0, num_tiles_block_c);
                }
                block_row_offset += block_row_size;
    // kernel_profiler::mark_time(6);
            }
            start_block_row_id += num_rows_block;
        }
        //DPRINT << 'W' << ENDL();
    }

    // kernel_profiler::mark_time(6);

}
