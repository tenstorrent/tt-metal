// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/moreh_softmax/kernels/common.hpp"

void generate_epsilon() {
    constexpr uint32_t cb_eps = 2;
    union {
        float f;
        uint32_t u;
    } u;
    u.u = get_arg_val<uint32_t>(5);
    cb_reserve_back(cb_eps, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_eps));
    for (int j = 0; j < 1024; j++) ptr[j] = uint16_t(0);

    for (int k = 0; k < 4; k += 2)
        for (int j = 0; j < 16; j++) ptr[k * 256 + j * 16] = uint16_t(u.u >> 16);
    cb_push_back(cb_eps, 1);
}

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);
    const uint32_t scaler = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_scaler = 1;

    const uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat input_data_format = get_dataformat(cb_id_in0);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t block_size = get_compile_time_arg_val(3);

    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

#ifdef FUSE_GAMMA
    constexpr uint32_t cb_id_gamma = 3;
    uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const DataFormat gamma_data_format = get_dataformat(cb_id_gamma);
    const InterleavedAddrGenFast<gamma_is_dram> gamm_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};
#endif

#ifdef FUSE_BETA
    constexpr uint32_t cb_id_beta = 4;
    uint32_t beta_addr = get_arg_val<uint32_t>(7);
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const DataFormat beta_data_format = get_dataformat(cb_id_beta);
    const InterleavedAddrGenFast<beta_is_dram> beta_addrg = {
        .bank_base_address = beta_addr, .page_size = beta_tile_bytes, .data_format = beta_data_format};
#endif

    // Generate constant tiles for layernorm compute
    generate_bcast_scaler(cb_id_scaler, scaler);
    generate_epsilon();

#ifdef DO_MASK_H
    // for mask_h
    constexpr uint32_t cb_id_mask_h = 5;
    const uint32_t mask_h = get_arg_val<uint32_t>(8);
    generate_mask_h(cb_id_mask_h, mask_h);
#endif

#ifdef DO_MASK_W
    // for mask_w
    constexpr uint32_t cb_id_mask_w = 6;
    const uint32_t mask_w = get_arg_val<uint32_t>(9);
    generate_mask_w(cb_id_mask_w, mask_w);
#endif

    uint32_t offs = 0;
    const uint32_t NCHt = num_rows_per_core;
    constexpr uint32_t onetile = 1;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
            cb_reserve_back(cb_id_in0, block_size);
            uint32_t input_l1_write_ptr = get_write_ptr(cb_id_in0);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_read_tile(offs + wt + r + tile_offset, input_addrg, input_l1_write_ptr);
                input_l1_write_ptr += input_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, block_size);
        }  // wt loop

        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
#ifdef FUSE_GAMMA
            cb_reserve_back(cb_id_gamma, block_size);
            uint32_t gamma_l1_write_addr = get_write_ptr(cb_id_gamma);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_read_tile(wt + r, gamm_addrg, gamma_l1_write_addr);
                gamma_l1_write_addr += gamma_tile_bytes;
            }  // block_size loop
            noc_async_read_barrier();
            cb_push_back(cb_id_gamma, block_size);
#endif

#ifdef FUSE_BETA
            cb_reserve_back(cb_id_beta, block_size);
            uint32_t beta_l1_write_addr = get_write_ptr(cb_id_beta);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_read_tile(wt + r, beta_addrg, beta_l1_write_addr);
                beta_l1_write_addr += beta_tile_bytes;
            }  // block_size loop
            noc_async_read_barrier();
            cb_push_back(cb_id_beta, block_size);
#endif
        }  // wt loop
        offs += Wt;
    }  // ncht loop
}  // void kernel_main()
