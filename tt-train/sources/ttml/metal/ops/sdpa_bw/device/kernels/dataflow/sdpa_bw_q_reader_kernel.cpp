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

template <typename AddrGen>
void read_row(
    const uint32_t start_idx,
    const uint32_t num_of_tiles,
    const uint32_t cb_id,
    const AddrGen& address_generator,
    const uint32_t tile_bytes) {
    cb_reserve_back(cb_id, num_of_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    for (uint32_t tile_idx = 0; tile_idx < num_of_tiles; ++tile_idx) {
        noc_async_read_tile(start_idx + tile_idx, address_generator, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_of_tiles);
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t grad_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);     // grad_output buffer address
    uint32_t attn_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);     // attn_output buffer address
    uint32_t query_addr = get_arg_val<uint32_t>(runtime_args_counter++);           // query buffer address
    uint32_t key_addr = get_arg_val<uint32_t>(runtime_args_counter++);             // key buffer address
    uint32_t value_addr = get_arg_val<uint32_t>(runtime_args_counter++);           // value buffer address
    uint32_t mask_addr = get_arg_val<uint32_t>(runtime_args_counter++);            // mask buffer address
    uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);   // intermediates buffer address
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);            // starting row for this core

    // Circular buffer indices
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;
    constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;
    constexpr uint32_t cb_query = tt::CBIndex::c_2;
    constexpr uint32_t cb_key = tt::CBIndex::c_3;
    constexpr uint32_t cb_value = tt::CBIndex::c_4;
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
    constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_7;
    constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_8;

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);              // query width in tiles
    constexpr uint32_t kWt = get_compile_time_arg_val(1);              // key/value width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(2);               // sequence length in tiles
    constexpr uint32_t block_size = get_compile_time_arg_val(3);       // block size
    constexpr uint32_t q_heads = get_compile_time_arg_val(4);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5);  // heads per group
    constexpr uint32_t num_of_batches = get_compile_time_arg_val(6);   // num of batches
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(7);      // sdpa scaler factor
    constexpr uint32_t minus_one_bits = get_compile_time_arg_val(8);   // used to transform mask from 1/0 to 0/-1
    constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(9);  // used to transform mask from 0/-1 to 0/-1e9F

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_output_args = TensorAccessorArgs<10>();
    constexpr auto attn_output_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto query_args = TensorAccessorArgs<attn_output_args.next_compile_time_args_offset()>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto intermediates_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1U;
    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;

    generate_tile_with_bfloat16_value(
        cb_reduction_scaler, one);  // generate tile with bfloat16 value 1.0 for reduction scaler
    // generate_matmul_row_reduce_tile(cb_matmul_reduce);  // generate tile for matmul row reduce
    generate_matmul_row_reduce_tile_fp32(cb_matmul_reduce);  // generate tile for matmul row reduce in fp32
    // print_tile(cb_matmul_reduce, 0);

    const uint32_t tile_bytes = get_tile_size(cb_grad_output);
    const DataFormat data_format = get_dataformat(cb_grad_output);

    // [DEBUG]: Use fp32 for intermediates to improve numerical stability
    // const uint32_t precise_tile_bytes = get_tile_size(cb_intermediates);
    // const DataFormat precise_data_format = get_dataformat(cb_intermediates);

    // Create TensorAccessor generators for inputs
    const auto grad_output_addr_generator = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto attn_output_addr_generator = TensorAccessor(attn_output_args, attn_output_addr, tile_bytes);
    const auto query_addr_generator = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key_addr_generator = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value_addr_generator = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto mask_addr_generator = TensorAccessor(mask_args, mask_addr, tile_bytes);
    // [DEBUG]: Use fp32 for intermediates to improve numerical stability
    // const auto intermediates_addr_generator =
    //     TensorAccessor(intermediates_args, intermediates_addr, precise_tile_bytes);
    const auto intermediates_addr_generator =
        TensorAccessor(intermediates_args, intermediates_addr, tile_bytes);

    const uint32_t num_of_groups = q_heads / heads_per_group;
    const uint32_t num_of_interm_tiles = 2U;

    const float scaler = uint32_to_float(scaler_bits);
    const float minus_one = uint32_to_float(minus_one_bits);
    const float custom_inf = uint32_to_float(custom_inf_bits);

    DPRINT << "SDPA BW Q: num_rows_to_process=" << num_rows_to_process << ", start_row=" << start_row << ", qWt=" << qWt
           << ", kWt=" << kWt << ", Ht=" << Ht << ", q_heads =" << q_heads << ", scaler=" << scaler
           << ", minus_one=" << minus_one << ", custom_inf=" << custom_inf << ENDL();

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t global_row_idx = start_row + i;
        uint32_t q_start_idx = global_row_idx * qWt;
        DPRINT << "Q_READER: Processing row " << i << ", global_row_idx=" << global_row_idx
               << ", q_start_idx=" << q_start_idx << ENDL();
        // Read query row
        read_row(q_start_idx, qWt, cb_query, query_addr_generator, tile_bytes);
        // Read attn_output row
        read_row(q_start_idx, qWt, cb_attn_output, attn_output_addr_generator, tile_bytes);
        // Read grad_output row
        read_row(q_start_idx, qWt, cb_grad_output, grad_output_addr_generator, tile_bytes);

        uint32_t q_head_idx = (global_row_idx / Ht) % q_heads;  // which head of Q we are processing right now

        // which batch we are processing right now
        uint32_t batch_idx = global_row_idx / (Ht * q_heads);
        // calculate which group of K and V we need to read for this head of Q
        uint32_t kv_group_idx = q_head_idx / heads_per_group;

        // calculate the starting index of K and V to read
        uint32_t kv_offset = (batch_idx * num_of_groups + kv_group_idx) * Ht * qWt;

        // calculate the starting index of attn_mask to read
        uint32_t mask_offset = (batch_idx * q_heads + q_head_idx) * Ht * Ht + (global_row_idx % Ht) * Ht;

        DPRINT << "Q_READER: batch_idx=" << batch_idx << ", q_head_idx=" << q_head_idx
               << ", kv_group_idx=" << kv_group_idx << ENDL();
        DPRINT << "Q_READER: kv_offset=" << kv_offset << ", mask_offset=" << mask_offset << ENDL();

        // read intermediates for current row of Q
        // intermediates shape: (B, qNH, S, 64) -> (batch, heads, seq_len, 2 tiles)
        uint32_t intermediates_idx = global_row_idx * num_of_interm_tiles;
        DPRINT << "Q_READER: intermediates_idx=" << intermediates_idx << ENDL();
        // [DEBUG]: Use fp32 for intermediates to improve numerical stability
        // read_row(
        //     intermediates_idx, num_of_interm_tiles, cb_intermediates, intermediates_addr_generator,
        //     precise_tile_bytes);
        read_row(
            intermediates_idx, num_of_interm_tiles, cb_intermediates, intermediates_addr_generator, tile_bytes);

        // cb_wait_front(cb_intermediates, num_of_interm_tiles);
        // print_tile(cb_intermediates, 0);
        // print_tile(cb_intermediates, 1);

        for (uint32_t h = 0; h < Ht; ++h) {
            uint32_t kv_start_idx =
                kv_offset + h * qWt;  // jump to the next row of K and V, qWt == kWt == vWt(same embedding size)

            if (i == 0 && h == 0) {
                DPRINT << "Q_READER: First K/V row - kv_start_idx=" << kv_start_idx << ENDL();
            }

            // Read one row of K and V
            read_row(kv_start_idx, qWt, cb_key, key_addr_generator, tile_bytes);
            read_row(kv_start_idx, qWt, cb_value, value_addr_generator, tile_bytes);

            // read one tile of attn_mask for current row of K and V
            // row of K define the column in (QK^T) matrix, so it define the column of attn_mask to read
            uint32_t mask_tile_idx = mask_offset + h;
            if (i == 0 && h == 0) {
                DPRINT << "Q_READER: mask_tile_idx=" << mask_tile_idx << ENDL();
            }

            cb_reserve_back(cb_attn_mask, onetile);
            uint32_t attn_mask_l1_write_addr = get_write_ptr(cb_attn_mask);
            noc_async_read_tile(mask_tile_idx, mask_addr_generator, attn_mask_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_attn_mask, onetile);
        }
    }
}
