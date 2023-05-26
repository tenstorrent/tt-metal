#include <stdint.h>
#include "dataflow_api.h"

uint64_t round_down_32(uint64_t a){
    return (a >> 5) << 5;
}

void kernel_main() {

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr                 = get_arg_val<uint32_t>(1);
    const uint32_t num_unpadded_W           = get_arg_val<uint32_t>(2);
    const uint32_t num_total_W              = get_arg_val<uint32_t>(3);
    const uint32_t num_unpadded_Z           = get_arg_val<uint32_t>(4);
    const uint32_t num_total_Z              = get_arg_val<uint32_t>(5);
    const uint32_t num_unpadded_Yt          = get_arg_val<uint32_t>(6);
    const uint32_t num_total_Yt             = get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_Xt          = get_arg_val<uint32_t>(8);
    const uint32_t num_total_Xt             = get_arg_val<uint32_t>(9);
    const uint32_t tile_size                = get_arg_val<uint32_t>(10);
    const uint32_t src_buffer_l1_addr       = get_arg_val<uint32_t>(11);


    #define tile_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (tile_size_is_pow2)
    const uint32_t log_base_2_of_page_size = get_arg_val<uint32_t>(12);
    const InterleavedPow2AddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    const InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const InterleavedAddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size
    };

    const InterleavedAddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .page_size = tile_size
    };
    #endif


    uint32_t src_tile_id = 0;
    uint32_t dst_tile_id = 0;
    for (uint32_t w = 0; w < num_total_W; w++) {
        for (uint32_t z = 0; z < num_total_Z; z++) {
            for (uint32_t yt = 0; yt < num_total_Yt; yt++) {
                for (uint32_t xt = 0; xt < num_total_Xt; xt++) {
                    //Unpad output
                    if (xt >= num_unpadded_Xt || yt >= num_unpadded_Yt || z >= num_unpadded_Z || w >= num_unpadded_W) {
                        src_tile_id++;
                    // Copy Input
                    } else {
                        uint64_t src_noc_addr = get_noc_addr(src_tile_id, s0);
                        noc_async_read(src_noc_addr, src_buffer_l1_addr, tile_size);
                        noc_async_read_barrier();
                        src_tile_id++;
                        uint64_t dst_noc_addr = get_noc_addr(dst_tile_id, s1);
                        noc_async_write(src_buffer_l1_addr, dst_noc_addr, tile_size);
                        noc_async_write_barrier();
                        dst_tile_id++;
                    }
                }
            }
        }
    }
}
