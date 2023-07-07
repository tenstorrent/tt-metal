#include <stdint.h>
#include "dataflow_kernel_api.h"

uint64_t round_down_32(uint64_t a){
    return (a >> 5) << 5;
}

void kernel_main() {

    constexpr uint32_t alignment = 32;

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr                 = get_arg_val<uint32_t>(1);
    const uint32_t num_unpadded_W           = get_arg_val<uint32_t>(2);
    const uint32_t num_total_W              = get_arg_val<uint32_t>(3);
    const uint32_t num_unpadded_Z           = get_arg_val<uint32_t>(4);
    const uint32_t num_total_Z              = get_arg_val<uint32_t>(5);
    const uint32_t num_unpadded_Y           = get_arg_val<uint32_t>(6);
    const uint32_t num_total_Y              = get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_X           = get_arg_val<uint32_t>(8);
    const uint32_t num_total_X              = get_arg_val<uint32_t>(9);
    const uint32_t unpadded_X_size          = get_arg_val<uint32_t>(10);
    const uint32_t padded_X_size            = get_arg_val<uint32_t>(11);
    const uint32_t padded_X_diff_size       = get_arg_val<uint32_t>(12);
    const uint32_t cache_buffer_l1_addr     = get_arg_val<uint32_t>(13);
    const uint32_t src_buffer_l1_addr       = get_arg_val<uint32_t>(14);
    const uint32_t dst_buffer_l1_addr       = get_arg_val<uint32_t>(15);


    std::uint32_t* cache_buffer = (uint32_t*)(cache_buffer_l1_addr);
    std::uint32_t* src_buffer = (uint32_t*)(src_buffer_l1_addr);
    std::uint32_t* dst_buffer = (uint32_t*)(dst_buffer_l1_addr);


    #define src_stick_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (src_stick_size_is_pow2)
    const uint32_t src_log_base_2_of_page_size = get_arg_val<uint32_t>(16);
    const dataflow::InterleavedPow2AddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .log_base_2_of_page_size = src_log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const dataflow::InterleavedAddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .page_size = padded_X_size
    };
    #endif

    #define dst_stick_size_is_pow2 get_compile_time_arg_val(1) == 1
    #if (dst_stick_size_is_pow2)
    const uint32_t dst_log_base_2_of_page_size = get_arg_val<uint32_t>(17);
    const dataflow::InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = dst_log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const dataflow::InterleavedAddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .page_size = unpadded_X_size
    };
    #endif

    uint32_t src_stick_id = 0;
    uint32_t dst_stick_id = 0;
    uint32_t l1_cache_addr = cache_buffer_l1_addr;

    uint32_t padded_Y_diff_rows = num_total_Y - num_unpadded_Y;
    uint32_t padded_Z_diff_rows = (num_total_Z - num_unpadded_Z) * num_total_Y;
    uint32_t padded_W_diff_rows = (num_total_W * num_unpadded_W) * num_total_Z * num_total_Y;

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t y = 0; y < num_unpadded_Y; y++) {
                uint64_t dst_noc_addr = dataflow::get_noc_addr(dst_stick_id, s1);
                uint64_t dst_round_down_addr = round_down_32(dst_noc_addr);
                uint32_t dst_diff_bytes = dst_noc_addr - dst_round_down_addr;
                uint32_t dst_buffer_l1_addr_real = dst_buffer_l1_addr + dst_diff_bytes;
                volatile std::uint32_t* dst = (volatile uint32_t*)(dst_buffer_l1_addr);

                uint64_t src_noc_addr = dataflow::get_noc_addr(
                    src_stick_id, s0);

                // Read from DRAM to src buffer
                uint64_t src_round_down_addr = round_down_32(src_noc_addr);
                uint64_t src_diff_bytes = src_noc_addr - src_round_down_addr;
                dataflow::noc_async_read(src_round_down_addr, src_buffer_l1_addr, unpadded_X_size + src_diff_bytes);

                // Copy from cache to dst buffer
                volatile std::uint32_t* cache = (volatile uint32_t*)(l1_cache_addr);
                for(uint32_t z = 0; z < dst_diff_bytes / 4; z++) {
                    dst[z] = cache[z];
                }

                dst = (volatile uint32_t*)(dst_buffer_l1_addr_real);

                // Block before copying data from src to dst buffer
                dataflow::noc_async_read_barrier();
                volatile std::uint32_t* data_buffer = (volatile uint32_t*)(src_buffer_l1_addr + src_diff_bytes);
                for(uint32_t z = 0; z < unpadded_X_size / 4; z++) {
                    dst[z] = data_buffer[z];
                }
                src_stick_id++;
                dataflow::noc_async_write(dst_buffer_l1_addr, dst_round_down_addr, unpadded_X_size + dst_diff_bytes);
                // Copy from tmp to cache
                uint64_t end_noc_addr = dst_noc_addr + unpadded_X_size;
                uint64_t end_round_down_addr = round_down_32(end_noc_addr);
                uint32_t cache_to_write = end_noc_addr - end_round_down_addr;
                dst = (volatile uint32_t*)(dst_buffer_l1_addr_real + unpadded_X_size - cache_to_write);
                for(uint32_t z = 0; z < (cache_to_write) / 4; z++) {
                    cache[z] = dst[z];
                }
                dst_stick_id++;
                if (dst_stick_id & 7) {
                    l1_cache_addr += alignment;
                } else {
                    l1_cache_addr = cache_buffer_l1_addr;
                }
                dataflow::noc_async_write_barrier();
            }
            src_stick_id += padded_Y_diff_rows;
        }
        src_stick_id += padded_Z_diff_rows;
    }
    // src_stick_id += padded_W_diff_rows;
}
