// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id  = get_arg_val<uint32_t>(2);
    uint32_t start_ht  = get_arg_val<uint32_t>(3);
    uint32_t start_wt  = get_arg_val<uint32_t>(4);


    uint32_t Ht  = get_arg_val<uint32_t>(5);
    uint32_t Wt  = get_arg_val<uint32_t>(6);
    uint32_t HtWt  = get_arg_val<uint32_t>(7);


    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    //uint32_t Ht  = (uint32_t)get_compile_time_arg_val(1);
    //uint32_t Wt  = (uint32_t)get_compile_time_arg_val(2);
    //uint32_t HtWt  = (uint32_t)get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t ht = start_ht;
    uint32_t wt = start_wt;
    uint32_t i_tile = start_id;

    // this reader will read a NHW tensor in NWH order
    for (uint32_t i = 0; i < num_tiles; i++){
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i_tile, s, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, onetile);
        i_tile += Wt; // stride in H
        ht += 1;
        if (ht == Ht) {
            ht = 0;
            i_tile += 1;
            wt += 1;
            if (wt == Wt) {
                wt = 0;
                i_tile -= Wt; // Start of next batch
            } else {
                i_tile -= HtWt; // Start of next col
            }
        }
    }
}
