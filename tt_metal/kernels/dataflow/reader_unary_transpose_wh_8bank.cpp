#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    // skip args 1,2,3 for compat with reader_unary, reader_unary_8bank
    uint32_t N = get_arg_val<uint32_t>(4); // args match the order of reader_unary
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    uint32_t i_tile_N = 0; // first tile in current batch
    uint32_t i_tile = 0;

    const InterleavedPow2AddrGen s = {
        .bank_base_address = src_addr,
        .num_used_banks = 8,
        .log_base_2_of_num_used_banks = 3,
        .log_base_2_of_bank_unit_size = 11
    };

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n<N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w<Wt; w++) {
            for (uint32_t h = 0; h<Ht; h++) {
                uint64_t src_noc_addr = get_noc_addr(i_tile, s);
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
                noc_async_read_barrier();

                cb_push_back(cb_id_in0, onetile);
                i_tile += Wt; // stride in H
            } // Ht
            i_tile -= HtWt; // go back to H=0
            i_tile += 1; // increment Wt
        } // Wt
        i_tile_N += HtWt; // stride in batch/channel
    } // N
}
