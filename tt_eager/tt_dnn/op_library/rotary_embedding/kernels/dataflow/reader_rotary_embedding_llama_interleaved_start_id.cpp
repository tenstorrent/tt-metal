// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
    //DPRINT << "Reader Hang 1" << ENDL();
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t cos_addr  = get_arg_val<uint32_t>(1);
    uint32_t sin_addr  = get_arg_val<uint32_t>(2);
    uint32_t trans_mat_addr = get_arg_val<uint32_t>(3);
    uint32_t num_rows = get_arg_val<uint32_t>(4); // Index correctly in the for loop
    uint32_t num_tiles_written = get_arg_val<uint32_t>(5); // Index correctly in the for loop
    uint32_t start_row_id = get_arg_val<uint32_t>(6); // Index correctly in the for loop
    uint32_t cos_sin_start_id = get_arg_val<uint32_t>(7); // Index correctly in the for loop

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(3);
    constexpr bool input_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool cos_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool sin_is_dram = get_compile_time_arg_val(6) == 1;
    constexpr bool trans_mat_is_dram = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t HtWt = get_compile_time_arg_val(10);

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const DataFormat input_data_format = get_dataformat(input_cb_id);

    const InterleavedAddrGenFast<input_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = input_tile_bytes,
        .data_format = input_data_format
    };

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const DataFormat cos_data_format = get_dataformat(cos_cb_id);

    const InterleavedAddrGenFast<cos_is_dram> s1 = {
        .bank_base_address = cos_addr,
        .page_size = cos_tile_bytes,
        .data_format = cos_data_format
    };

    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const DataFormat sin_data_format = get_dataformat(sin_cb_id);

    const InterleavedAddrGenFast<sin_is_dram> s2 = {
        .bank_base_address = sin_addr,
        .page_size = sin_tile_bytes,
        .data_format = sin_data_format
    };

    const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb_id);
    const DataFormat trans_mat_format = get_dataformat(trans_mat_cb_id);

    const InterleavedAddrGenFast<trans_mat_is_dram> s3 = {
        .bank_base_address = trans_mat_addr,
        .page_size = trans_mat_tile_bytes,
        .data_format = trans_mat_format
    };

    uint32_t input_curr_id = num_tiles_written;
    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t trans_mat_curr_id = 0;
    uint32_t ht = start_row_id;

    // Read transformation matrix in CB (only once, because it will be reused)
    cb_reserve_back(trans_mat_cb_id, onetile);
    uint32_t trans_mat_l1_write_addr = get_write_ptr(trans_mat_cb_id);
    noc_async_read_tile(trans_mat_curr_id, s3, trans_mat_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(trans_mat_cb_id, onetile);

    /*
        Read a ublock of tiles from src to CB, and then push the ublock to unpacker

        num_rows = 1 * 8 * 128 * 128 // 128 // 32 = 32
        Ht = 4
        Wt = 4
    */
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {

            // Read input into CB
            cb_reserve_back(input_cb_id, onetile);
            uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
            noc_async_read_tile(input_curr_id, s0, input_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(input_cb_id, onetile);
            input_curr_id++;

            // Read sin into CB
            cb_reserve_back(sin_cb_id, onetile);
            uint32_t sin_l1_write_addr = get_write_ptr(sin_cb_id);
            noc_async_read_tile(cos_sin_curr_id, s2, sin_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(sin_cb_id, onetile);

            // Read cos into CB
            cb_reserve_back(cos_cb_id, onetile);
            uint32_t cos_l1_write_addr = get_write_ptr(cos_cb_id);
            noc_async_read_tile(cos_sin_curr_id, s1, cos_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cos_cb_id, onetile);

            cos_sin_curr_id++;
        }

        /*
            sin and cos matrices are duplicated across num_heads. So, reset their indices
            here to duplicate them into CB

        */
        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
    }


}
