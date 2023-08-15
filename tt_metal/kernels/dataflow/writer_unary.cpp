#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out0 = 16;

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);

    // uint32_t ublock_size_bytes = 64;

    uint32_t ublock_size_tiles = 1;

    kernel_profiler::mark_time(5);
    // kernel_profiler::mark_time(6);
    // kernel_profiler::mark_time(7);
    // kernel_profiler::mark_time(8);
    // kernel_profiler::mark_time(9);
    // kernel_profiler::mark_time(10);
    // kernel_profiler::mark_time(11);
    // kernel_profiler::mark_time(12);
    // kernel_profiler::mark_time(13);
    // kernel_profiler::mark_time(14);
    // kernel_profiler::mark_time(15);
    // kernel_profiler::mark_time(16);

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);

            // kernel_profiler::mark_time(5);
        cb_wait_front(cb_id_out0, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

            kernel_profiler::mark_time(7);
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

            // kernel_profiler::mark_time(7);
        noc_async_write_barrier();

            // kernel_profiler::mark_time(8);
        cb_pop_front(cb_id_out0, ublock_size_tiles);
            // kernel_profiler::mark_time(9);
        dst_addr += ublock_size_bytes;
    }

    kernel_profiler::mark_time(6);
}
