// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api_addrgen.h>
#include <hostdevcommon/kernel_structs.h>

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

namespace NAMESPACE {

void read_block_tiles(
    const uint32_t cb_input_idx,
    const InterleavedAddrGenFast<true>& input_address_generator,
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t tile_bytes,
    const uint32_t idx) {
    for (uint32_t j = 0; j < Wt; j += block_size) {
        cb_reserve_back(cb_input_idx, block_size);
        uint32_t l1_write_addr = get_write_ptr(cb_input_idx);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            noc_async_read_tile(idx + j + block_idx, input_address_generator, l1_write_addr);
            l1_write_addr += tile_bytes;
        }

        noc_async_read_barrier();
        cb_push_back(cb_input_idx, block_size);
    }
}

void read_coloumn_tiles() {
}

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t query_address = get_arg_val<uint32_t>(runtime_args_counter++);        // query buffer address
    uint32_t key_address = get_arg_val<uint32_t>(runtime_args_counter++);          // key buffer address
    uint32_t value_address = get_arg_val<uint32_t>(runtime_args_counter++);        // value buffer address
    uint32_t mask_address = get_arg_val<uint32_t>(runtime_args_counter++);         // mask buffer address
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t packed_scaler = get_compile_time_arg_val(3);

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const DataFormat data_format = get_dataformat(cb_query);

    const InterleavedAddrGenFast</* is_dram */ true> query_address_generator = {
        .bank_base_address = query_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> key_address_generator = {
        .bank_base_address = key_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> value_address_generator = {
        .bank_base_address = value_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> mask_address_generator = {
        .bank_base_address = mask_address, .page_size = tile_bytes, .data_format = data_format};

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    // TODO: check if I need the cb with scaler, or using mul_unary_tile(cb, scaler) will be better
    generate_tile_with_packed_bfloat16_value(cb_scaler, packed_scaler);
    generate_tile_with_bfloat16_value(
        cb_reduction_scaler, one);  // generate tile with bfloat16 value 1.0 for reduction scaler

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t idx = (start_row + i) * Wt;

        for (uint32_t col = 0; col < Wt; col += block_size) {
            // read query block_size tiles
            cb_reserve_back(cb_query, block_size);
            uint32_t query_l1_write_addr = get_write_ptr(cb_query);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                noc_async_read_tile(idx + col + block_idx, query_address_generator, query_l1_write_addr);
                query_l1_write_addr += tile_bytes;
            }

            noc_async_read_barrier();
            cb_push_back(cb_query, block_size);

            for (uint32_t j = 0; j < Wt; j += block_size) {
                // read key row block_size tiles
                // we read key by rows to compute matmul Q by K^T
                cb_reserve_back(cb_key, block_size);
                uint32_t key_l1_writer_addr = get_write_ptr(cb_key);
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                    noc_async_read_tile(idx + j + block_idx, key_address_generator, key_l1_writer_addr);
                    key_l1_writer_addr += tile_bytes;
                }

                noc_async_read_barrier();
                cb_push_back(cb_key, block_size);
            }
            // for k_block, v_blcok in K, V
            //      read k_block
            //      read q_block <- nee to read it in transpose way: read col instead of row
        }
    }
}

}  // namespace NAMESPACE
