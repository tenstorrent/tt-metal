#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t col_start_tile_id = get_arg_val<uint32_t>(1); // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg_val<uint32_t>(2);
    uint32_t num_cols = get_arg_val<uint32_t>(3); // number of cols to read

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t Ht  = get_compile_time_arg_val(1);
    constexpr uint32_t Wt  = get_compile_time_arg_val(2);
    constexpr uint32_t HtWt  = get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    #ifdef REDUCE_SCALER
    constexpr uint32_t cb_in_2 = 2;
    constexpr uint32_t scaler = get_compile_time_arg_val(4);
    cb_reserve_back(cb_in_2, 1);
    if (scaler != 0) {
        uint16_t u = uint16_t(scaler>>16);
        auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));
        for (int j = 0; j < 1024; j++)
            ptr[j] = uint16_t(0);

        for (int k = 0; k < 4; k++)
        for (int j = 0; j < 16; j++)
            ptr[k*256 + j] = u;

    }
    cb_push_back(cb_in_2, 1);
    #endif

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t w = curr_col_in_batch;

    // this reader will read a NHW tensor in NWH order
    for (uint32_t i = 0; i < num_cols; i++) {
        uint32_t curr_id = col_start_tile_id;
        for (uint32_t j = 0; j < Ht; j++) {
            cb_reserve_back(cb_id_in0, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile(curr_id, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            curr_id += Wt; // stride in H
        }
        w++;
        if (w == Wt) {
            col_start_tile_id = curr_id - Wt + 1;
            w = 0;
        } else {
            col_start_tile_id++;
        }
    }
}
