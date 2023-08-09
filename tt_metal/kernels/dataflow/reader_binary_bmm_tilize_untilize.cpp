#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t src0_num_blocks  = get_arg_val<uint32_t>(3);
    uint32_t src1_addr  = get_arg_val<uint32_t>(4);
    uint32_t src1_noc_x = get_arg_val<uint32_t>(5);
    uint32_t src1_noc_y = get_arg_val<uint32_t>(6);
    uint32_t src1_num_blocks  = get_arg_val<uint32_t>(7);
    // uint32_t ublock_size_tiles  = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

    // uint32_t ublock_size_bytes_0 = 64;
    // uint32_t ublock_size_bytes_1 = 64;
    uint32_t ublock_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t num_blocks = src0_num_blocks > src1_num_blocks ? src0_num_blocks : src1_num_blocks;

    kernel_profiler::mark_time(5);


    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i=0; i<num_blocks; i += ublock_size_tiles) {
        // if (i < src0_num_blocks) {
            uint64_t src0_noc_addr = get_noc_addr(src0_noc_x, src0_noc_y, src0_addr);
            uint64_t src1_noc_addr = get_noc_addr(src1_noc_x, src1_noc_y, src1_addr);

            // kernel_profiler::mark_time(5);
            cb_reserve_back(cb_id_in0, ublock_size_tiles);
            cb_reserve_back(cb_id_in1, ublock_size_tiles);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);

            // kernel_profiler::mark_time(6);
            noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
            noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

            // kernel_profiler::mark_time(7);
            noc_async_read_barrier();

            // kernel_profiler::mark_time(8);
            cb_push_back(cb_id_in0, ublock_size_tiles);
            cb_push_back(cb_id_in1, ublock_size_tiles);
            // kernel_profiler::mark_time(9);

            src0_addr += ublock_size_bytes_0;
            src1_addr += ublock_size_bytes_1;
        // }

        // if (i < src1_num_blocks) {
        //     uint64_t src1_noc_addr = get_noc_addr(src1_noc_x, src1_noc_y, src1_addr);

        //     // kernel_profiler::mark_time(10);
        //     cb_reserve_back(cb_id_in1, ublock_size_tiles);
        //     l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        //     // kernel_profiler::mark_time(11);
        //     noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

        //     // kernel_profiler::mark_time(12);
        //     noc_async_read_barrier();

        //     // kernel_profiler::mark_time(13);
        //     cb_push_back(cb_id_in1, ublock_size_tiles);
        //     // kernel_profiler::mark_time(14);

        //     src1_addr += ublock_size_bytes_1;
        // }
    }

    kernel_profiler::mark_time(6);
}
