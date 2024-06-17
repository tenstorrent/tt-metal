// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto mean_addr = get_arg_val<uint32_t>(1);
    const auto rstd_addr = get_arg_val<uint32_t>(2);
    const auto num_rows_per_core = get_arg_val<uint32_t>(3);
    const auto Wt = get_arg_val<uint32_t>(4);
    const auto tile_offset = get_arg_val<uint32_t>(5);

    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool mean_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool rstd_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(3) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t block_size = get_compile_time_arg_val(5);

    constexpr uint32_t cb_id_output = tt::CB::c_out0;
    constexpr uint32_t cb_id_mean = tt::CB::c_out1;
    constexpr uint32_t cb_id_rstd = tt::CB::c_out2;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    // mean
    const uint32_t mean_tile_bytes = get_tile_size(cb_id_mean);
    const auto mean_data_format = get_dataformat(cb_id_mean);

    const InterleavedAddrGenFast<mean_is_dram> mean_addrg = {
        .bank_base_address = mean_addr, .page_size = mean_tile_bytes, .data_format = mean_data_format};

    // rstd
    const uint32_t rstd_tile_bytes = get_tile_size(cb_id_rstd);
    const auto rstd_data_format = get_dataformat(cb_id_rstd);

    const InterleavedAddrGenFast<rstd_is_dram> rstd_addrg = {
        .bank_base_address = rstd_addr, .page_size = rstd_tile_bytes, .data_format = rstd_data_format};

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;
    const auto NCHt = num_rows_per_core;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        if (mean_has_value) {
            // mean
            const auto mean_l1_read_addr = get_read_ptr(cb_id_mean);
            cb_wait_front(cb_id_mean, onetile);
            noc_async_write_tile((offs + tile_offset) / Wt, mean_addrg, mean_l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_mean, onetile);
        }  // mean_has_value

        if (rstd_has_value) {
            // rstd
            const auto rstd_l1_read_addr = get_read_ptr(cb_id_rstd);
            cb_wait_front(cb_id_rstd, onetile);
            noc_async_write_tile((offs + tile_offset) / Wt, rstd_addrg, rstd_l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_rstd, onetile);
        }  // rstd_has_value

        // output
        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
            cb_wait_front(cb_id_output, block_size);
            auto output_l1_read_addr = get_read_ptr(cb_id_output);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_write_tile(offs + wt + r + tile_offset, output_addrg, output_l1_read_addr);
                output_l1_read_addr += output_tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_output, block_size);
        }  // wt loop

        offs += Wt;
    }  // ncht loop
}  // void kernel_main()
