// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

template <typename AddrGen>
void read_head(
    const uint32_t start_idx,
    const uint32_t num_of_tiles,
    const uint32_t cb_id,
    const AddrGen& address_generator,
    const uint32_t tile_bytes) {
    cb_reserve_back(cb_id, num_of_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    for (uint32_t tile_idx = 0; tile_idx < num_of_tiles; ++tile_idx) {
        // TODO[improve](vmelnykov): this method is deprecated
        // need to use noc_async_read_page FORCE_INLINE void noc_async_read_page(...)
        noc_async_read_tile(start_idx + tile_idx, address_generator, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_of_tiles);
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
    constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;

    constexpr uint32_t qWt = get_compile_time_arg_val(0);  // num tile in inner dim in query (d/TILE_W)
    constexpr uint32_t kWt = get_compile_time_arg_val(1);  // num tile in inner dim in key/value (d/TILE_W)
    constexpr uint32_t Ht = get_compile_time_arg_val(2);   // (S / TILE_H)
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t q_heads = get_compile_time_arg_val(4);          // num of heads in query
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5);  // num of heads per group
    constexpr uint32_t num_of_batches = get_compile_time_arg_val(6);   // num of batches
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(7);      // sdpa scaler factor
    constexpr uint32_t minus_one_bits = get_compile_time_arg_val(8);   // used to transform mask from 1/0 to 0/-1
    constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(9);  // used to transform mask from 0/-1 to 0/-1e9F
    constexpr auto query_args = TensorAccessorArgs<10>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();

#ifdef USE_ATTN_MASK
    constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
#endif

    constexpr uint32_t onetile = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const DataFormat data_format = get_dataformat(cb_query);

    const auto query_address_generator = TensorAccessor(query_args, query_address, tile_bytes);
    const auto key_address_generator = TensorAccessor(key_args, key_address, tile_bytes);
    const auto value_address_generator = TensorAccessor(value_args, value_address, tile_bytes);

#ifdef USE_ATTN_MASK
    const auto mask_address_generator = TensorAccessor(mask_args, mask_address, tile_bytes);
#endif

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;

    generate_tile_with_bfloat16_value(
        cb_reduction_scaler, one);                      // generate tile with bfloat16 value 1.0 for reduction scaler
    generate_matmul_row_reduce_tile(cb_matmul_reduce);  // generate tile for matmul row reduce

    const float scaler = uint32_to_float(scaler_bits);
    const float minus_one = uint32_to_float(minus_one_bits);
    const float custom_inf = uint32_to_float(custom_inf_bits);

    const uint32_t num_of_groups = q_heads / heads_per_group;

    DPRINT << "SDPA FW: num_rows_to_process=" << num_rows_to_process << ", start_row=" << start_row << ", qWt=" << qWt
           << ", kWt=" << kWt << ", Ht=" << Ht << ", q_heads =" << q_heads << ", scaler=" << scaler
           << ", minus_one=" << minus_one << ", custom_inf=" << custom_inf << ENDL();

    // while we process one q_chunk (head of Q), we stream all K and V chunks (heads of K and V)
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t global_row_idx = start_row + i;
        uint32_t q_start_idx = global_row_idx * qWt;
        read_head(q_start_idx, qWt, cb_query, query_address_generator, tile_bytes);

        uint32_t q_head_idx = (global_row_idx / Ht) % q_heads;  // which head of Q we are processing right now

        // which batch we are processing right now
        uint32_t batch_idx = global_row_idx / (Ht * q_heads);
        // calculate which group of K and V we need to read for this head of Q
        uint32_t kv_group_idx = q_head_idx / heads_per_group;

        uint32_t kv_offset =
            (batch_idx * num_of_groups + kv_group_idx) * qWt * Ht;  // jump to start of relevant batch/group of K and V

        uint32_t mask_offset = (batch_idx * q_heads + q_head_idx) * Ht * Ht + (global_row_idx % Ht) * Ht;

        for (uint32_t h = 0; h < Ht; ++h) {
            uint32_t kv_start_idx = kv_offset + h * qWt;  // jump to the next row
            read_head(kv_start_idx, qWt, cb_key, key_address_generator, tile_bytes);

#ifdef USE_ATTN_MASK
            // read one tile of attn_mask for current row of K and V
            // row of K define the column in (QK^T) matrix, so it define the column of attn_mask
            cb_reserve_back(cb_attn_mask, onetile);
            uint32_t attn_mask_l1_writer_addr = get_write_ptr(cb_attn_mask);
            noc_async_read_tile(mask_offset + h, mask_address_generator, attn_mask_l1_writer_addr);
            noc_async_read_barrier();
            cb_push_back(cb_attn_mask, onetile);
#endif
            read_head(kv_start_idx, qWt, cb_value, value_address_generator, tile_bytes);
        }
    }
}
