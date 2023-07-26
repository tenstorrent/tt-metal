#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    const uint32_t dst_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t num_unpadded_W           = get_arg_val<uint32_t>(1);
    const uint32_t num_padded_Wt            = get_arg_val<uint32_t>(2);
    const uint32_t num_unpadded_Z           = get_arg_val<uint32_t>(3);
    const uint32_t num_padded_Zt            = get_arg_val<uint32_t>(4);
    const uint32_t num_unpadded_Yt          = get_arg_val<uint32_t>(5);
    const uint32_t num_padded_Yt            = get_arg_val<uint32_t>(6);
    const uint32_t num_unpadded_Xt          = get_arg_val<uint32_t>(7);
    const uint32_t num_padded_Xt            = get_arg_val<uint32_t>(8);
    const uint32_t pad_value                = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_out0                            = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out1                            = get_compile_time_arg_val(1);
    constexpr bool dst_is_dram                               = get_compile_time_arg_val(2) == 1;

    const uint32_t tile_size = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram> s1 = {
        .bank_base_address = dst_addr,
        .page_size = tile_size,
        .data_format = data_format
    };

    cb_reserve_back(cb_id_out1, 1); // in this kernel we are not pushing anything into CBs, just using the space

    uint32_t pad_buffer_l1_addr = get_write_ptr(cb_id_out1);

    // Fill pad tile with pad value
    volatile tt_l1_ptr uint32_t* pad_buffer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_buffer_l1_addr);
    const uint32_t num_elems = tile_size / sizeof(uint32_t);
    for (uint32_t z = 0; z < num_elems; z++) {
        pad_buffer[z] = pad_value;
    }

    uint32_t src_tile_id = 0;
    uint32_t dst_tile_id = 0;

    auto pad_tiles = [&] (uint32_t num_tiles) {
        for(uint32_t pad_tile = 0; pad_tile < num_tiles; pad_tile++) {
            noc_async_write_tile(dst_tile_id, s1, pad_buffer_l1_addr);
            dst_tile_id++;
        }
        noc_async_write_barrier();
    };

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t yt = 0; yt < num_unpadded_Yt; yt++) {
                for (uint32_t xt = 0; xt < num_unpadded_Xt; xt++) {
                    cb_wait_front(cb_id_out0, 1);
                    uint32_t src_buffer_l1_addr = get_read_ptr(cb_id_out0);
                    noc_async_write_tile(dst_tile_id, s1, src_buffer_l1_addr);
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_out0, 1);
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
