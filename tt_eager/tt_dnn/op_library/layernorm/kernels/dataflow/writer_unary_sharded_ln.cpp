// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

FORCE_INLINE void generate_bcast_scaler_c() {
    constexpr uint32_t cb_in_4 = tt::CB::c_in4;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(0);
    cb_reserve_back(cb_in_4, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_4));

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[(k << 8) + j] = uint16_t(u.u>>16);
    cb_push_back(cb_in_4, 1);
}

FORCE_INLINE void generate_bcast_scaler_w() {
    constexpr uint32_t cb_in_2 = tt::CB::c_in2;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_in_2, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_in_2));

    for (int k = 0; k < 4; k++)
    for (int j = 0; j < 16; j++)
        ptr[(k << 8) + j] = uint16_t(u.u>>16);
    cb_push_back(cb_in_2, 1);
}

FORCE_INLINE void generate_epsilon() {
    constexpr uint32_t eps_cb_id = tt::CB::c_in3;
    union { float f; uint32_t u; } u; u.u = get_arg_val<uint32_t>(2);
    cb_reserve_back(eps_cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(eps_cb_id));

    for (int k = 0; k < 4; k+=2)
    for (int j = 0; j < 16; j++)
        ptr[(k << 8) + (j << 4)] = uint16_t(u.u>>16);
    cb_push_back(eps_cb_id, 1);
}

void kernel_main() {
    constexpr bool is_all_to_all_worker              = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma                       = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta                        = get_compile_time_arg_val(2) == 1;
    constexpr bool gamma_is_dram                    = get_compile_time_arg_val(3) == 1;
    constexpr bool beta_is_dram                     = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t block_w                      = get_compile_time_arg_val(5);

    const uint32_t gamma_addr                     = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr                      = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id            = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id             = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;


    generate_bcast_scaler_w();
    if constexpr(is_all_to_all_worker) {
        generate_bcast_scaler_c();
    }
    generate_epsilon();

    if constexpr(fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const DataFormat gamma_data_format = get_dataformat(cb_gamma);
        const InterleavedAddrGenFast<gamma_is_dram> gamma = {
            .bank_base_address = gamma_addr,
            .page_size = gamma_tile_bytes,
            .data_format = gamma_data_format
        };

        uint32_t l1_write_addr_gamma = get_write_ptr(cb_gamma);
        cb_reserve_back(cb_gamma, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            noc_async_read_tile(tile_id, gamma, l1_write_addr_gamma);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, block_w);
    }

    if constexpr(fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        const DataFormat beta_data_format = get_dataformat(cb_beta);
        const InterleavedAddrGenFast<beta_is_dram> beta = {
            .bank_base_address = beta_addr,
            .page_size = beta_tile_bytes,
            .data_format = beta_data_format
        };

        uint32_t l1_write_addr_beta = get_write_ptr(cb_beta);
        cb_reserve_back(cb_beta, block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            noc_async_read_tile(tile_id, beta, l1_write_addr_beta);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_beta, block_w);
    }

}
