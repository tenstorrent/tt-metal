#include <cstdint>

void kernel_main() {
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t dram_dst_noc_x        = get_arg_val<uint32_t>(5);
    std::uint32_t dram_dst_noc_y        = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

    // int k = 0;
    // for (k=5;k<17;k++) kernel_profiler::mark_time(k);

    bool flag_0 = true;
    bool flag_1 = true;
    bool flag_2 = true;
    bool flag_3 = true;
    bool flag_4 = true;
    bool flag_5 = true;
    bool flag_6 = true;
    bool flag_7 = true;
    bool flag_8 = true;
    bool flag_9 = true;
    bool flag_10 = true;
    bool flag_11 = true;

    // kernel_profiler::mark_time_once(5, &flag_0);
    // kernel_profiler::mark_time_once(6, &flag_1);
    // kernel_profiler::mark_time_once(7, &flag_2);
    // kernel_profiler::mark_time_once(8, &flag_3);
    // kernel_profiler::mark_time_once(9, &flag_4);
    // kernel_profiler::mark_time_once(10, &flag_5);
    // kernel_profiler::mark_time_once(11, &flag_6);
    // kernel_profiler::mark_time_once(12, &flag_7);
    // kernel_profiler::mark_time_once(13, &flag_8);
    // kernel_profiler::mark_time_once(14, &flag_9);
    // kernel_profiler::mark_time_once(15, &flag_10);
    // kernel_profiler::mark_time_once(16, &flag_11);

    kernel_profiler::mark_time(5);
    kernel_profiler::mark_time(6);
    kernel_profiler::mark_time(7);
    kernel_profiler::mark_time(8);
    kernel_profiler::mark_time(9);
    kernel_profiler::mark_time(10);
    kernel_profiler::mark_time(11);
    kernel_profiler::mark_time(12);
    kernel_profiler::mark_time(13);
    kernel_profiler::mark_time(14);
    kernel_profiler::mark_time(15);
    kernel_profiler::mark_time(16);

    // kernel_profiler::mark_time(5);
    std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);
    // kernel_profiler::mark_time(6);
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
    // kernel_profiler::mark_time(7);
    noc_async_read_barrier();

    // kernel_profiler::mark_time(8);
    std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);
    // kernel_profiler::mark_time(9);
    noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
    // kernel_profiler::mark_time(10);
    noc_async_write_barrier();
}
