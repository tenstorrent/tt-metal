#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    // skip args 1,2,3 for compat with reader_unary, reader_unary_8bank
    uint32_t N = get_arg_val<uint32_t>(4); // args match the order of reader_unary
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t first_tile = get_arg_val<uint32_t>(8);
    uint32_t Wt_partitioned = get_arg_val<uint32_t>(9);
    uint32_t Ht_partitioned = get_arg_val<uint32_t>(10);
    uint32_t HtpWt = get_arg_val<uint32_t>(11);
    uint32_t scaler = get_arg_val<uint32_t>(12);
    if (scaler != 0) {
        union { float f; uint32_t u; } u; u.u = scaler;
        //DPRINT << "TWH part Scaler = " << F32(u.f) << ENDL();
        constexpr uint32_t cb_in_2 = 2;
        dataflow::cb_reserve_back(cb_in_2, 1);
        auto ptr = reinterpret_cast<uint16_t*>(dataflow::get_write_ptr(cb_in_2));
        for (int j = 0; j < 1024; j++)
            ptr[j] = uint16_t(0);

        for (int k = 0; k < 4; k++)
        for (int j = 0; j < 16; j++)
            ptr[k*256 + j] = uint16_t(u.u>>16);
        dataflow::cb_push_back(cb_in_2, 1);
    }

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    uint32_t i_tile_N = first_tile; // first tile in current batch
    uint32_t i_tile = 0;

    const dataflow::InterleavedPow2AddrGen<true> s = {
        .bank_base_address = src_addr,


        .log_base_2_of_page_size = 11
    };

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n<N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w<Wt_partitioned; w++) {
            for (uint32_t h = 0; h<Ht_partitioned; h++) {
                uint64_t src_noc_addr = dataflow::get_noc_addr(i_tile, s);
                dataflow::cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = dataflow::get_write_ptr(cb_id_in0);
                dataflow::noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
                dataflow::noc_async_read_barrier();

                dataflow::cb_push_back(cb_id_in0, onetile);
                i_tile += Wt; // stride in H
            } // Ht
            i_tile -= HtpWt; // go back to start H
            i_tile += 1; // increment Wt
        } // Wt
        i_tile_N += HtWt; // stride in batch/channel
    } // N
}
