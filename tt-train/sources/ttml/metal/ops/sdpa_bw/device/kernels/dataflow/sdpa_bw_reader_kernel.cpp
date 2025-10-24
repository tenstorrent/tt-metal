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
    uint32_t grad_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);     // grad_output buffer address
    uint32_t attn_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);     // attn_output buffer address
    uint32_t query_addr = get_arg_val<uint32_t>(runtime_args_counter++);           // query buffer address
    uint32_t key_addr = get_arg_val<uint32_t>(runtime_args_counter++);             // key buffer address
    uint32_t value_addr = get_arg_val<uint32_t>(runtime_args_counter++);           // value buffer address
    uint32_t mask_addr = get_arg_val<uint32_t>(runtime_args_counter++);            // mask buffer address
    uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);   // intermediates buffer address
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    uint32_t start_row =
        get_arg_val<uint32_t>(runtime_args_counter++);  // pre calculated num_rows_written in program factory

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

    // [DEBUG]: Used for debug, should be removed later
    constexpr auto cb_masked_interm = tt::CBIndex::c_20;

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
    constexpr uint32_t num_of_interm_tiles = 2U;

    const uint32_t tile_bytes = get_tile_size(cb_grad_output);
    const DataFormat data_format = get_dataformat(cb_grad_output);

    // Create TensorAccessor generators
    const auto grad_output_address_generator = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto attn_output_address_generator = TensorAccessor(attn_output_args, attn_output_addr, tile_bytes);
    const auto query_address_generator = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key_address_generator = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value_address_generator = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto mask_address_generator = TensorAccessor(mask_args, mask_addr, tile_bytes);
    const auto intermediates_address_generator = TensorAccessor(intermediates_args, intermediates_addr, tile_bytes);

    constexpr uint16_t one = 0x00003F80;                          // (bfloat16)1.0 -> uint16_t
    generate_tile_with_bfloat16_value(cb_reduction_scaler, one);  // generate tile with bfloat16 value 1.0
    generate_matmul_row_reduce_tile(cb_matmul_reduce);            // generate tile for matmul row reduce

    const float scaler = uint32_to_float(scaler_bits);
    const float minus_one = uint32_to_float(minus_one_bits);
    const float custom_inf = uint32_to_float(custom_inf_bits);

    const uint32_t num_of_groups = q_heads / heads_per_group;

    DPRINT << "SDPA BW: num_rows_to_process=" << num_rows_to_process << ", start_row=" << start_row << ", qWt=" << qWt
           << ", kWt=" << kWt << ", Ht=" << Ht << ", q_heads =" << q_heads << ", scaler=" << scaler
           << ", minus_one=" << minus_one << ", custom_inf=" << custom_inf << ENDL();

    // process rows of K and V assigned to this core
    // stream rows from Q, dO, O(for all heads associated with this group of K and V)
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t global_row_idx = start_row + i;
        uint32_t kv_start_idx = global_row_idx * kWt;

        DPRINT << "Reader: Processing row " << i << ", global_row_idx=" << global_row_idx
               << ", kv_start_idx=" << kv_start_idx << ENDL();

        read_row(kv_start_idx, kWt, cb_key, key_address_generator, tile_bytes);
        read_row(kv_start_idx, kWt, cb_value, value_address_generator, tile_bytes);

        uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;  // which group of K and V we are processing
        uint32_t batch_idx = global_row_idx / (Ht * num_of_groups);  // which batch we are processing

        // the index of the first head in Q associated with this group of K and V
        uint32_t first_q_head_idx = group_idx * heads_per_group;
        uint32_t q_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * qWt;

        DPRINT << "Reader: batch_idx=" << batch_idx << ", group_idx=" << group_idx
               << ", first_q_head_idx=" << first_q_head_idx << ", q_offset=" << q_offset << ENDL();

        // the offset of attn_mask associated with this group of K and V
        // jump to relevent batch and head, then jump to the row in attn_mask associated with current row of K and V
        uint32_t mask_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * Ht + (global_row_idx % Ht);

        uint32_t intermediates_offset = (batch_idx * q_heads + first_q_head_idx) * Ht;

        // TODO: add calculation for dO, O indexes because in forward pass they are stored with shape (B, 1, S,
        // qNH*qEmbd)
        for (uint32_t q_head_idx = 0; q_head_idx < heads_per_group; ++q_head_idx) {
            for (uint32_t h = 0; h < Ht; ++h) {
                uint32_t q_start_idx = q_offset + (q_head_idx * Ht + h) * qWt;
                read_row(q_start_idx, qWt, cb_query, query_address_generator, tile_bytes);
                read_row(q_start_idx, qWt, cb_grad_output, grad_output_address_generator, tile_bytes);
                read_row(q_start_idx, qWt, cb_attn_output, attn_output_address_generator, tile_bytes);

                // read one tile of attn_mask for current row of K and V
                // row of K define the column in (QK^T) matrix, so it define the column of attn_mask
                cb_reserve_back(cb_attn_mask, onetile);
                uint32_t attn_mask_l1_writer_addr = get_write_ptr(cb_attn_mask);
                noc_async_read_tile(mask_offset + h * Ht, mask_address_generator, attn_mask_l1_writer_addr);
                noc_async_read_barrier();
                cb_push_back(cb_attn_mask, onetile);

                // Read intermediates - one tile per row (contains 1/sum_exp values from forward pass)
                // TODO[improve](vmelnykov): Now we share two intermediates values per head row: row-wise max value and
                // 1/sum_exp In future we can think about optimizing this by sharing logsumexp only
                uint32_t intermediates_idx = intermediates_offset + h * num_of_interm_tiles;
                read_row(
                    intermediates_idx,
                    num_of_interm_tiles,
                    cb_intermediates,
                    intermediates_address_generator,
                    tile_bytes);

                // [DEBUG]: Used for debug, should be removed later
                // cb_wait_front(cb_masked_interm, num_of_interm_tiles);
                // print_tile(cb_masked_interm, 0);
                // print_tile(cb_masked_interm, 1);
                // print_tile(cb_intermediates, 0);
                // print_tile(cb_intermediates, 1);
            }
            // update offsets to point to the next head in attn_mask and intermediates
            mask_offset += Ht * Ht;  // jump to the next head in attn_mask associated with current Q head
            intermediates_offset += Ht * num_of_interm_tiles;  // jump to the head in intermediates
        }
    }
}
