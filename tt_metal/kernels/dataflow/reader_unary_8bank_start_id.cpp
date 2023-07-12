#include <stdint.h>
#include "dataflow_api.h"

//#include "debug_print.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(3); // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank
    uint32_t start_id = get_arg_val<uint32_t>(4);
    uint32_t scaler = get_arg_val<uint32_t>(5);
    if (scaler != 0) {
        union { float f; uint32_t u; } u; u.u = scaler;
        //DPRINT << "startid Scaler = " << F32(u.f) << ENDL();
        constexpr uint32_t cb_in_2 = 2;
        cb_reserve_back(cb_in_2, 1);
        auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));
        for (int j = 0; j < 1024; j++)
            ptr[j] = uint16_t(0);

        for (int k = 0; k < 4; k++)
        for (int j = 0; j < 16; j++)
            ptr[k*256 + j] = uint16_t(u.u>>16);
        cb_push_back(cb_in_2, 1);
    }

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = src_addr,


        .log_base_2_of_page_size = 11
    };

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i_tile = 0;
    for (uint32_t i = start_id; i<start_id + num_tiles; i ++) {
        uint64_t src_noc_addr = get_noc_addr(i, s);
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
