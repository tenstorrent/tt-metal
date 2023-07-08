#include <stdint.h>
#include "dataflow_api.h"

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
    const uint32_t pad_value                = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t tile_size = get_tile_size(cb_id_in0);

    constexpr DataFormat data_format                      = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr bool src0_is_dram                           = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_is_dram                            = get_compile_time_arg_val(2) == 1;
    // In and out are assumed to be same dataformat
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = data_format
    };

    const InterleavedAddrGenFast<dst_is_dram> s1 = {
        .bank_base_address = dst_addr,
        .page_size = tile_size,
        .data_format = data_format
    };

    cb_reserve_back(cb_id_in0, 1); // in this kernel we are not pushing anything into CBs, just using the space
    cb_reserve_back(cb_id_in1, 1); // in this kernel we are not pushing anything into CBs, just using the space

    uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    uint32_t pad_buffer_l1_addr = get_write_ptr(cb_id_in1);

    // Fill pad tile with pad value
    volatile uint32_t* pad_buffer = reinterpret_cast<volatile uint32_t*>(pad_buffer_l1_addr);
    const uint32_t num_elems = tile_size / sizeof(uint32_t);
    for (uint32_t z = 0; z < num_elems; z++) {
        pad_buffer[z] = pad_value;
    }

    uint32_t src_tile_id = 0;
    uint32_t dst_tile_id = 0;

    auto pad_tiles = [&] (uint32_t num_tiles) {
        for(uint32_t pad_tile = 0; pad_tile < num_tiles; pad_tile++) {
            noc_async_write_tile(dst_tile_id, s1, pad_buffer_l1_addr);
            noc_async_write_barrier();
            dst_tile_id++;
        }
    };

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t yt = 0; yt < num_unpadded_Yt; yt++) {
                for (uint32_t xt = 0; xt < num_unpadded_Xt; xt++) {
                    noc_async_read_tile(src_tile_id, s0, src_buffer_l1_addr);
                    noc_async_read_barrier();
                    src_tile_id++;
                    noc_async_write_tile(dst_tile_id, s1, src_buffer_l1_addr);
                    noc_async_write_barrier();
                    dst_tile_id++;
                }
                pad_tiles(num_padded_Xt);
            }
            pad_tiles(num_padded_Yt);
        }
        pad_tiles(num_padded_Zt);
    }
    pad_tiles(num_padded_Wt);
}
