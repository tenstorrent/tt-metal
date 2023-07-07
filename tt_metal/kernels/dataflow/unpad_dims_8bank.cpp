#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr                 = get_arg_val<uint32_t>(1);
    const uint32_t num_unpadded_W           = get_arg_val<uint32_t>(2);
    const uint32_t num_padded_Wt            = get_arg_val<uint32_t>(3);
    const uint32_t num_unpadded_Z           = get_arg_val<uint32_t>(4);
    const uint32_t num_padded_Zt            = get_arg_val<uint32_t>(5);
    const uint32_t num_unpadded_Yt          = get_arg_val<uint32_t>(6);
    const uint32_t num_padded_Yt            = get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_Xt          = get_arg_val<uint32_t>(8);
    const uint32_t num_padded_Xt            = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t tile_size = get_tile_size(cb_id_in0);

    #define tile_size_is_pow2 get_compile_time_arg_val(0) == 1
    #if (tile_size_is_pow2)
    const uint32_t log_base_2_of_page_size = get_arg_val<uint32_t>(10);
    const dataflow::InterleavedPow2AddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    const dataflow::InterleavedPow2AddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size // TODO(AP): refactor
    };
    #else
    const dataflow::InterleavedAddrGen<true> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size
    };

    const dataflow::InterleavedAddrGen<true> s1 = {
        .bank_base_address = dst_addr,
        .page_size = tile_size
    };
    #endif

    dataflow::cb_reserve_back(cb_id_in0, 1); // in this kernel we are not pushing anything into CBs, just using the space

    uint32_t src_buffer_l1_addr = dataflow::get_write_ptr(cb_id_in0);

    uint32_t src_tile_id = 0;
    uint32_t dst_tile_id = 0;

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t yt = 0; yt < num_unpadded_Yt; yt++) {
                for (uint32_t xt = 0; xt < num_unpadded_Xt; xt++) {
                    // Copy Input
                    uint64_t src_noc_addr = dataflow::get_noc_addr(src_tile_id, s0);
                    dataflow::noc_async_read(src_noc_addr, src_buffer_l1_addr, tile_size);
                    dataflow::noc_async_read_barrier();
                    src_tile_id++;
                    uint64_t dst_noc_addr = dataflow::get_noc_addr(dst_tile_id, s1);
                    dataflow::noc_async_write(src_buffer_l1_addr, dst_noc_addr, tile_size);
                    dataflow::noc_async_write_barrier();
                    dst_tile_id++;
                }
                src_tile_id += num_padded_Xt;
            }
            src_tile_id += num_padded_Yt;
        }
        src_tile_id += num_padded_Zt;
    }
    // src_tile_id += num_padded_Wt;
}
