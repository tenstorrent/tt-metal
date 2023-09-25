// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void generate_bcast_scaler() {
    constexpr uint32_t cb_in_2 = tt::CB::c_in2;
    uint32_t scaler = get_arg_val<uint32_t>(4);
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
    constexpr uint32_t cb_mask = tt::CB::c_in1;
    int mask_w = static_cast<int>(get_arg_val<uint32_t>(5));
    union { float f; uint32_t u; } one; one.f = 1.0f;
    union { float f; uint32_t u; } zero; zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask));

    for (int j = 0; j < 1024; j++)
        ptr[j] = uint16_t(0);

    for(int h = 0 ; h < 16; h++) {
        // sub tile 0
        {
            int mask_w_0 = mask_w;
            if (mask_w_0 >= 16) mask_w_0 = 16;
            int w = 0;
            for(; w < mask_w_0; w++){
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            int mask_w_1 = mask_w - 16;
            if (mask_w_1 < 0) mask_w_1 = 0;
            int w = 0;
            for(; w < mask_w_1; w++){
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            int mask_w_0 = mask_w;
            if (mask_w_0 >= 16) mask_w_0 = 16;
            int w = 0;
            for(; w < mask_w_0; w++){
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            int mask_w_1 = mask_w - 16;
            if (mask_w_1 < 0) mask_w_1 = 0;
            int w = 0;
            for(; w < mask_w_1; w++){
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for(; w < 16; w++){
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}

void kernel_main() {

    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);

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
    generate_mask();

    // read ublocks from src0 to CB0, then push ublocks to compute (unpacker)
    uint32_t curr_tile = tile_offset;
    for (uint32_t i=0; i< N; i += onetile) {
        for(uint32_t w = 0 ; w < Wt; w++){
            cb_reserve_back(cb_in, onetile);
            l1_write_addr_in = get_write_ptr(cb_in);
            noc_async_read_tile(curr_tile, src_in, l1_write_addr_in);
            noc_async_read_barrier();
            cb_push_back(cb_in, onetile);
            curr_tile++;
        }
    }
}
