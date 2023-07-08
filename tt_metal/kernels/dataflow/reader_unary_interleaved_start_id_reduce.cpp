#include <stdint.h>
#include "dataflow_api.h"


void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t scaler = get_arg_val<uint32_t>(3);
    if (scaler != 0) {
        constexpr uint32_t cb_in_2 = 2;
        cb_reserve_back(cb_in_2, 1);
        auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));
        for (int j = 0; j < 1024; j++)
            ptr[j] = uint16_t(0);

        for (int k = 0; k < 4; k++)
        for (int j = 0; j < 16; j++)
            ptr[k*256 + j] = uint16_t(scaler>>16);
        cb_push_back(cb_in_2, 1);
    }

    constexpr DataFormat data_format = static_cast<DataFormat>(get_compile_time_arg_val(0));
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = start_id; i<start_id + num_tiles; i ++) {
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
