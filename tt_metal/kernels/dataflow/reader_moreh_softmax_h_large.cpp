// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = tt::CB::c_in2;
    uint32_t scaler = get_arg_val<uint32_t>(5);
    union { float f; uint32_t u; } u; u.u = scaler;
    cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));

    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[k*256 + j] = uint16_t(u.u>>16);
    cb_push_back(cb_in_2, 1);
}

void kernel_main() {

    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = tt::CB::c_in0;

    uint32_t l1_write_addr_in;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src_in_tile_bytes = get_tile_size(cb_in);
    const DataFormat src_in_data_format = get_dataformat(cb_in);

    constexpr bool in_is_dram = get_compile_time_arg_val(0) == 1;

    const InterleavedAddrGenFast<in_is_dram> src_in = {
        .bank_base_address = src_addr,
        .page_size = src_in_tile_bytes,
        .data_format = src_in_data_format
    };

    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();

    // read ublocks from src0 to CB0, then push ublocks to compute (unpacker)
    uint32_t curr_tile = tile_offset;
    for (uint32_t i=0; i< N; i += onetile) {
        uint32_t w_idx = curr_tile % Wt;
        uint32_t nc_idx = curr_tile / Wt;
        uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        for(uint32_t h = 0 ; h < Ht; h++){
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += Wt;
        }

        w_idx = curr_tile % Wt;
        nc_idx = curr_tile / Wt;
        tile_idx = nc_idx * Ht * Wt + w_idx;
        for(uint32_t h = 0 ; h < Ht; h++){
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(tile_idx, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            tile_idx += Wt;
        }

        curr_tile += 1;
    }
}
