// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = tt::CB::c_in2;
    uint32_t scaler = get_arg_val<uint32_t>(6);
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

void generate_mask() {
    constexpr uint32_t cb_mask = tt::CB::c_in3;
    int mask_h = static_cast<int>(get_arg_val<uint32_t>(7));
    union { float f; uint32_t u; } one; one.f = 1.0f;
    union { float f; uint32_t u; } zero; zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask));

    for(int w = 0; w < 16; w++) {
        // sub tile 0
        {
            int mask_h_0 = mask_h;
            if (mask_h_0 >= 16) mask_h_0 = 16;
            int h = 0;
            for(; h < mask_h_0; h++){
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            int mask_h_0 = mask_h;
            if (mask_h_0 >= 16) mask_h_0 = 16;
            int h = 0;
            for(; h < mask_h_0; h++){
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            int mask_h_1 = mask_h - 16;
            if (mask_h_1 < 0) mask_h_1 = 0;
            int h = 0;
            for(; h < mask_h_1; h++){
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            int mask_h_1 = mask_h - 16;
            if (mask_h_1 < 0) mask_h_1 = 0;
            int h = 0;
            for(; h < mask_h_1; h++){
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for(; h < 16; h++){
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}

void kernel_main() {
    uint32_t y_addr  = get_arg_val<uint32_t>(0);
    uint32_t dy_addr  = get_arg_val<uint32_t>(1);

    uint32_t N = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);
    uint32_t Ht = get_arg_val<uint32_t>(4);
    uint32_t Wt = get_arg_val<uint32_t>(5);

    constexpr auto cb_y = tt::CB::c_in0;
    constexpr auto cb_dy = tt::CB::c_in1;

    uint32_t l1_write_addr_in;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t y_tile_bytes = get_tile_size(cb_y);
    const DataFormat y_data_format = get_dataformat(cb_y);

    uint32_t dy_tile_bytes = get_tile_size(cb_dy);
    const DataFormat dy_data_format = get_dataformat(cb_dy);

    constexpr bool y_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dy_is_dram = get_compile_time_arg_val(1) == 1;

    const InterleavedAddrGenFast<y_is_dram> y_in = {
        .bank_base_address = y_addr,
        .page_size = y_tile_bytes,
        .data_format = y_data_format
    };

    const InterleavedAddrGenFast<dy_is_dram> dy_in = {
        .bank_base_address = dy_addr,
        .page_size = dy_tile_bytes,
        .data_format = dy_data_format
    };

    // TODO(AP): cleanup, probably with named args/param pack/reflection.
    generate_bcast_scaler();
    generate_mask();

    // read ublocks from src0 to CB0, then push ublocks to compute (unpacker)
    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile)
    {
        uint32_t w_idx = curr_tile % Wt;
        uint32_t nc_idx = curr_tile / Wt;
        uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++)
        {
            // read y
            cb_reserve_back(cb_y, onetile);
            l1_write_addr_in = get_write_ptr(cb_y);
            noc_async_read_tile(tile_idx, y_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_y, onetile);

            // read dy
            cb_reserve_back(cb_dy, onetile);
            l1_write_addr_in = get_write_ptr(cb_dy);
            noc_async_read_tile(tile_idx, dy_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_dy, onetile);

            tile_idx += Wt;
        }

        w_idx = curr_tile % Wt;
        nc_idx = curr_tile / Wt;
        tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++)
        {
            // read y
            cb_reserve_back(cb_y, onetile);
            l1_write_addr_in = get_write_ptr(cb_y);
            noc_async_read_tile(tile_idx, y_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_y, onetile);

            // read dy
            cb_reserve_back(cb_dy, onetile);
            l1_write_addr_in = get_write_ptr(cb_dy);
            noc_async_read_tile(tile_idx, dy_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_dy, onetile);

            tile_idx += Wt;
        }
        curr_tile += 1;
    }
}
