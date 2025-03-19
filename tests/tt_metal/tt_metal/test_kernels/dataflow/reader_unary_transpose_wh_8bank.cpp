// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    // skip args 1,2,3 for compat with reader_unary, reader_unary_8bank
    uint32_t N = get_arg_val<uint32_t>(4);  // args match the order of reader_unary
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Wt = get_arg_val<uint32_t>(6);
    uint32_t HtWt = get_arg_val<uint32_t>(7);
    uint32_t scaler = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    if (scaler != 0) {
        union {
            float f;
            uint32_t u;
        } u;
        u.u = scaler;
        // DPRINT << "TWH Scaler = " << F32(u.f) << ENDL();
        constexpr uint32_t cb_in_2 = 2;
        cb_reserve_back(cb_in_2, 1);
        auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));
        for (int j = 0; j < 1024; j++) {
            ptr[j] = uint16_t(0);
        }

        for (int k = 0; k < 4; k++) {
            for (int j = 0; j < 16; j++) {
                ptr[k * 256 + j] = uint16_t(u.u >> 16);
            }
        }
        cb_push_back(cb_in_2, 1);
    }

    uint32_t i_tile_N = 0;  // first tile in current batch
    uint32_t i_tile = 0;

    const InterleavedPow2AddrGen<true> s = {
        .bank_base_address = src_addr,

        .log_base_2_of_page_size = 11};

    // this reader will read a NHW tensor in NWH order
    for (uint32_t n = 0; n < N; n++) {
        i_tile = i_tile_N;
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                uint64_t src_noc_addr = get_noc_addr(i_tile, s);
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read(src_noc_addr, l1_write_addr, tile_bytes);
                noc_async_read_barrier();

                cb_push_back(cb_id_in0, onetile);
                i_tile += Wt;  // stride in H
            }  // Ht
            i_tile -= HtWt;  // go back to H=0
            i_tile += 1;     // increment Wt
        }  // Wt
        i_tile_N += HtWt;  // stride in batch/channel
    }  // N
}
