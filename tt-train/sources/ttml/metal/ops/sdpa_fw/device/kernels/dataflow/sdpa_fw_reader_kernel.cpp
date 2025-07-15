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

void read_coloumn_tiles() {
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
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

    //[DEBUG] TODO: remove this
    constexpr uint32_t cb_transpose_key = tt::CBIndex::c_6;

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // d / TILE_W
    constexpr uint32_t Ht = get_compile_time_arg_val(2);  // S / TILE_W
    constexpr uint32_t packed_scaler = get_compile_time_arg_val(3);

    const uint32_t kv_chunks_number = Ht;
    const uint32_t q_chunk_size = Wt;
    const uint32_t k_chunk_size = Wt;
    const uint32_t v_chunk_size = Wt;
    const uint32_t kv_chunks_size = Wt;

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const DataFormat data_format = get_dataformat(cb_query);

    const InterleavedAddrGenFast</* is_dram */ true> query_address_generator = {
        .bank_base_address = query_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> key_address_generator = {
        .bank_base_address = key_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> value_address_generator = {
        .bank_base_address = value_address, .page_size = tile_bytes, .data_format = data_format};

    // const InterleavedAddrGenFast</* is_dram */ true> mask_address_generator = {
    //     .bank_base_address = mask_address, .page_size = tile_bytes, .data_format = data_format};

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    // TODO: check if I need the cb with scaler, or using mul_unary_tile(cb, scaler) will be better
    generate_tile_with_packed_bfloat16_value(cb_scaler, packed_scaler);
    generate_tile_with_bfloat16_value(
        cb_reduction_scaler, one);  // generate tile with bfloat16 value 1.0 for reduction scaler

    // while we process one q_chunk (row of Q), we stream all K and V chunks (rows of K and V)
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t idx = (start_row + i) * q_chunk_size;

        // TODO[improve]: change offset calculation to support reading rows from different batches
        uint32_t key_page_offset = ((start_row + i) / Ht) * Wt * Ht;

        cb_reserve_back(cb_query, q_chunk_size);  // 12 head
        uint32_t query_l1_write_addr = get_write_ptr(cb_query);
        for (uint32_t col = 0; col < q_chunk_size; ++col) {
            noc_async_read_tile(idx + col, query_address_generator, query_l1_write_addr);
            query_l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_query, q_chunk_size);

        // stream K and V by rows while processing Q row
        for (uint32_t h = 0; h < kv_chunks_number; ++h) {  // read all
            uint32_t k_idx =
                key_page_offset + h * kv_chunks_size;  // add row offset in case I need to read second batch
            // read key row block_size tiles
            // we read key by rows to compute matmul Q by K^T
            cb_reserve_back(cb_key, kv_chunks_size);
            cb_reserve_back(cb_value, kv_chunks_size);
            uint32_t key_l1_writer_addr = get_write_ptr(cb_key);
            uint32_t value_l1_writer_addr = get_write_ptr(cb_value);
            for (uint32_t w = 0; w < kv_chunks_size; ++w) {  // reading row of K and V
                noc_async_read_tile(k_idx + w, key_address_generator, key_l1_writer_addr);
                noc_async_read_tile(k_idx + w, value_address_generator, value_l1_writer_addr);
                key_l1_writer_addr += tile_bytes;
                value_l1_writer_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_key, kv_chunks_size);
            cb_push_back(cb_value, kv_chunks_size);
        }
    }
}
