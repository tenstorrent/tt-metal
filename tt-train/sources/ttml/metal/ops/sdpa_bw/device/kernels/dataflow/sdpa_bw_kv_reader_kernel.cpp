// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <hostdevcommon/kernel_structs.h>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#ifdef CHAIN
#include <hostdevcommon/common_values.hpp>

#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t query_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t key_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t value_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t u_scaler_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;
    constexpr uint32_t cb_query = tt::CBIndex::c_2;
    constexpr uint32_t cb_key = tt::CBIndex::c_3;
    constexpr uint32_t cb_value = tt::CBIndex::c_4;
#ifdef USE_ATTN_MASK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
#endif
    constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_7;
    constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_14;

    constexpr uint32_t qWt = get_compile_time_arg_val(0);              // Q/K width in tiles
    constexpr uint32_t kWt = get_compile_time_arg_val(1);              // key width in tiles (K read)
    constexpr uint32_t vWt = get_compile_time_arg_val(2);              // V/dO/O width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(3);               // sequence length in tiles
    constexpr uint32_t q_heads = get_compile_time_arg_val(4);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(5);  // heads per group

    // compile-time args 6..9: chain-mode semaphores and chunk rows
    constexpr auto grad_output_args = TensorAccessorArgs<12>();  // 6..11: chain semaphores and chunk rows
    constexpr auto query_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto intermediates_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto u_scaler_args = TensorAccessorArgs<intermediates_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1U;
    constexpr uint32_t num_of_interm_tiles = 1U;

    const uint32_t tile_bytes = get_tile_size(cb_grad_output);
    const uint32_t interm_tile_bytes = get_tile_size(cb_intermediates);

    const auto grad_output_address_generator = TensorAccessor(grad_output_args, grad_output_addr);
    const auto query_address_generator = TensorAccessor(query_args, query_addr);
    const auto key_address_generator = TensorAccessor(key_args, key_addr);
    const auto value_address_generator = TensorAccessor(value_args, value_addr);
#ifdef USE_ATTN_MASK
    const auto mask_address_generator = TensorAccessor(mask_args, mask_addr);
#endif
    const auto intermediates_address_generator =
        TensorAccessor(intermediates_args, intermediates_addr, interm_tile_bytes);
    const auto u_scaler_address_generator = TensorAccessor(u_scaler_args, u_scaler_addr);

    generate_matmul_row_reduce_tile(cb_matmul_reduce);

    const uint32_t num_of_groups = q_heads / heads_per_group;

#ifdef CHAIN
    // Chain mode (head split): an item is (K/V row, query head); its own K/V row is read here, the stream's
    // Q / lse / dO / u rows (query head of the stream, all Ht query tiles) arrive in chunks of chain_chunk rows:
    // read from DRAM by the injector, received from the previous core otherwise, forwarded to the next core
    // by every core but the sink.
    // The writer forwards every chunk to the next core on the other NoC (iteration 23): this reader hands it
    // chunk c with received = c + 1 and, before reusing a chunk's CB slots, waits for forwarded >= c - 1.
    constexpr uint32_t chain_chunk = get_compile_time_arg_val(9);
    Semaphore received_sem(get_compile_time_arg_val(10));
    Semaphore forwarded_sem(get_compile_time_arg_val(11));
    const uint32_t prev_noc_x = num_rows_to_process;
    const uint32_t prev_noc_y = start_row;
    runtime_args_counter += 2U;  // next core's NoC coordinates: the writer forwards
    const uint32_t is_injector = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t is_sink = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t q_stream = get_arg_val<uint32_t>(runtime_args_counter++);  // batch * q_heads + query head
    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t items_base = runtime_args_counter;

    Semaphore sender_sem(get_compile_time_arg_val(6));
    Semaphore receiver_sem(get_compile_time_arg_val(7));
    const Noc noc;

    const uint32_t q_offset = q_stream * Ht * qWt;
    const uint32_t vo_offset = q_stream * Ht * vWt;
    const uint32_t interm_offset = q_stream * Ht * num_of_interm_tiles;
    const uint32_t u_offset = q_stream * Ht;
    constexpr uint32_t q_chunk_tiles = chain_chunk * qWt;
    constexpr uint32_t vo_chunk_tiles = chain_chunk * vWt;
    constexpr uint32_t one_chunk_tiles = chain_chunk;
    constexpr uint32_t chunks_per_round = Ht / chain_chunk;

    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t item = get_arg_val<uint32_t>(items_base + round);
        if (item != 0xFFFFFFFFU) {
            const uint32_t global_row_idx = item / heads_per_group;
            read_tiles_by_row(cb_key, key_address_generator, global_row_idx * kWt, kWt, tile_bytes, kWt);
            read_tiles_by_row(cb_value, value_address_generator, global_row_idx * vWt, vWt, tile_bytes, vWt);
        }
        for (uint32_t h0 = 0; h0 < Ht; h0 += chain_chunk) {
            const uint32_t c = round * chunks_per_round + h0 / chain_chunk;
            if (!is_sink && c >= 2U) {
                forwarded_sem.wait_min(c - 1U);  // the slots' previous chunk (c - 2) has been forwarded
            }
            cb_reserve_back(cb_query, q_chunk_tiles);
            cb_reserve_back(cb_intermediates, one_chunk_tiles);
            cb_reserve_back(cb_grad_output, vo_chunk_tiles);
            cb_reserve_back(cb_u_scalar_row, one_chunk_tiles);
            const uint32_t q_addr = get_write_ptr(cb_query);
            const uint32_t i_addr = get_write_ptr(cb_intermediates);
            const uint32_t vo_addr = get_write_ptr(cb_grad_output);
            const uint32_t u_addr = get_write_ptr(cb_u_scalar_row);
            if (is_injector) {
                // The chunk's Q / lse / dO / u tile reads all in flight, one barrier (iteration 22: four barriers
                // per row, serial with the forwarding, were a large part of the injector's time).
                uint32_t qa = q_addr;
                uint32_t ia = i_addr;
                uint32_t voa = vo_addr;
                uint32_t ua = u_addr;
                const uint32_t u_tile_bytes = get_tile_size(cb_u_scalar_row);
                for (uint32_t h = h0; h < h0 + chain_chunk; ++h) {
                    for (uint32_t t = 0; t < qWt; ++t) {
                        noc_async_read_page(q_offset + h * qWt + t, query_address_generator, qa);
                        qa += tile_bytes;
                    }
                    for (uint32_t t = 0; t < num_of_interm_tiles; ++t) {
                        noc_async_read_page(
                            interm_offset + h * num_of_interm_tiles + t, intermediates_address_generator, ia);
                        ia += interm_tile_bytes;
                    }
                    for (uint32_t t = 0; t < vWt; ++t) {
                        noc_async_read_page(vo_offset + h * vWt + t, grad_output_address_generator, voa);
                        voa += tile_bytes;
                    }
                    noc_async_read_page(u_offset + h, u_scaler_address_generator, ua);
                    ua += u_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_query, q_chunk_tiles);
                cb_push_back(cb_intermediates, one_chunk_tiles);
                cb_push_back(cb_grad_output, vo_chunk_tiles);
                cb_push_back(cb_u_scalar_row, one_chunk_tiles);
            } else {
                receiver_sem.set(INVALID);
                sender_sem.up(noc, prev_noc_x, prev_noc_y, 1);
                receiver_sem.wait(VALID);
                cb_push_back(cb_query, q_chunk_tiles);
                cb_push_back(cb_intermediates, one_chunk_tiles);
                cb_push_back(cb_grad_output, vo_chunk_tiles);
                cb_push_back(cb_u_scalar_row, one_chunk_tiles);
            }
            received_sem.set(c + 1U);  // the writer forwards it
        }
    }
    noc.async_atomic_barrier();
#elif defined(WORK_LIST)
    // Work-list mode: an item is (K/V row, first query head, head count); with HEAD_SPLIT one head per item.
#ifdef HEAD_SPLIT
    constexpr uint32_t items_per_row = heads_per_group;
    constexpr uint32_t heads_per_item = 1U;
#else
    constexpr uint32_t items_per_row = 1U;
    constexpr uint32_t heads_per_item = heads_per_group;
#endif
    const uint32_t list_base = runtime_args_counter - 1U;
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t item = get_arg_val<uint32_t>(list_base + i);
        const uint32_t global_row_idx = item / items_per_row;
        const uint32_t head_begin = (item % items_per_row) * heads_per_item;

        read_tiles_by_row(cb_key, key_address_generator, global_row_idx * kWt, kWt, tile_bytes, kWt);
        read_tiles_by_row(cb_value, value_address_generator, global_row_idx * vWt, vWt, tile_bytes, vWt);

        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;
        const uint32_t batch_idx = global_row_idx / (Ht * num_of_groups);
        const uint32_t first_q_head_idx = group_idx * heads_per_group + head_begin;
        const uint32_t q_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * qWt;
        const uint32_t vo_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * vWt;
        const uint32_t k_row_tile = global_row_idx % Ht;
        const uint32_t num_q_tiles_to_read = Ht - k_row_tile;  // causal
        uint32_t intermediates_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * num_of_interm_tiles;

        for (uint32_t q_head_idx = 0; q_head_idx < heads_per_item; ++q_head_idx) {
            const uint32_t u_scaler_head_offset = (batch_idx * q_heads + first_q_head_idx + q_head_idx) * Ht;
            for (uint32_t q_idx = 0; q_idx < num_q_tiles_to_read; ++q_idx) {
                const uint32_t h = k_row_tile + q_idx;
                read_tiles_by_row(
                    cb_query, query_address_generator, q_offset + (q_head_idx * Ht + h) * qWt, qWt, tile_bytes, qWt);
                read_tiles_by_row(
                    cb_intermediates,
                    intermediates_address_generator,
                    intermediates_offset + h * num_of_interm_tiles,
                    num_of_interm_tiles,
                    interm_tile_bytes,
                    num_of_interm_tiles);
                read_tiles_by_row(
                    cb_grad_output,
                    grad_output_address_generator,
                    vo_offset + (q_head_idx * Ht + h) * vWt,
                    vWt,
                    tile_bytes,
                    vWt);
                read_one_tile(cb_u_scalar_row, u_scaler_address_generator, u_scaler_head_offset + h);
            }
            intermediates_offset += Ht * num_of_interm_tiles;
        }
    }
#elif defined(BALANCED_PARALLELISM)
    constexpr uint32_t pairs_per_seq = Ht / 2;

    auto read_row = [&](const uint32_t global_row_idx) {
        const uint32_t k_start_idx = global_row_idx * kWt;
        const uint32_t v_start_idx = global_row_idx * vWt;

        read_tiles_by_row(cb_key, key_address_generator, k_start_idx, kWt, tile_bytes, kWt);
        read_tiles_by_row(cb_value, value_address_generator, v_start_idx, vWt, tile_bytes, vWt);

        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;
        const uint32_t batch_idx = global_row_idx / (Ht * num_of_groups);

        const uint32_t first_q_head_idx = group_idx * heads_per_group;
        const uint32_t q_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * qWt;
        const uint32_t vo_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * vWt;

        const uint32_t k_row_tile = global_row_idx % Ht;

        const uint32_t q_start_tile = k_row_tile;
        const uint32_t num_q_tiles_to_read = Ht - k_row_tile;

        uint32_t intermediates_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * num_of_interm_tiles;

        for (uint32_t q_head_idx = 0; q_head_idx < heads_per_group; ++q_head_idx) {
            const uint32_t u_scaler_head_offset = (batch_idx * q_heads + first_q_head_idx + q_head_idx) * Ht;

            for (uint32_t q_idx = 0; q_idx < num_q_tiles_to_read; ++q_idx) {
                const uint32_t h = q_start_tile + q_idx;

                const uint32_t q_start_idx = q_offset + (q_head_idx * Ht + h) * qWt;
                read_tiles_by_row(cb_query, query_address_generator, q_start_idx, qWt, tile_bytes, qWt);

                uint32_t intermediates_idx = intermediates_offset + h * num_of_interm_tiles;
                read_tiles_by_row(
                    cb_intermediates,
                    intermediates_address_generator,
                    intermediates_idx,
                    num_of_interm_tiles,
                    interm_tile_bytes,
                    num_of_interm_tiles);

                const uint32_t vo_start_idx = vo_offset + (q_head_idx * Ht + h) * vWt;
                read_tiles_by_row(cb_grad_output, grad_output_address_generator, vo_start_idx, vWt, tile_bytes, vWt);

                read_one_tile(cb_u_scalar_row, u_scaler_address_generator, u_scaler_head_offset + h);
            }
            intermediates_offset += Ht * num_of_interm_tiles;
        }
    };

    for (uint32_t p = 0; p < num_rows_to_process; ++p) {
        const uint32_t global_pair_idx = start_row + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        read_row(heavy_global_row);
        read_row(light_global_row);
    }
#else
    // process rows of K and V assigned to this core
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t global_row_idx = start_row + i;
        const uint32_t k_start_idx = global_row_idx * kWt;
        const uint32_t v_start_idx = global_row_idx * vWt;

        read_tiles_by_row(cb_key, key_address_generator, k_start_idx, kWt, tile_bytes, kWt);
        read_tiles_by_row(cb_value, value_address_generator, v_start_idx, vWt, tile_bytes, vWt);

        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;
        const uint32_t batch_idx = global_row_idx / (Ht * num_of_groups);

        const uint32_t first_q_head_idx = group_idx * heads_per_group;
        const uint32_t q_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * qWt;
        const uint32_t vo_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * vWt;

        const uint32_t k_row_tile = global_row_idx % Ht;

#ifdef CAUSAL_MASK
        const uint32_t q_start_tile = k_row_tile;
        const uint32_t num_q_tiles_to_read = Ht - k_row_tile;
#else
        const uint32_t q_start_tile = 0;
        const uint32_t num_q_tiles_to_read = Ht;
#endif

#ifdef USE_ATTN_MASK
        const uint32_t mask_offset = k_row_tile;
#endif

        uint32_t intermediates_offset = (batch_idx * q_heads + first_q_head_idx) * Ht * num_of_interm_tiles;

        for (uint32_t q_head_idx = 0; q_head_idx < heads_per_group; ++q_head_idx) {
            const uint32_t u_scaler_head_offset = (batch_idx * q_heads + first_q_head_idx + q_head_idx) * Ht;

            for (uint32_t q_idx = 0; q_idx < num_q_tiles_to_read; ++q_idx) {
                const uint32_t h = q_start_tile + q_idx;

                const uint32_t q_start_idx = q_offset + (q_head_idx * Ht + h) * qWt;
                read_tiles_by_row(cb_query, query_address_generator, q_start_idx, qWt, tile_bytes, qWt);

#ifdef USE_ATTN_MASK
                read_one_tile(cb_attn_mask, mask_address_generator, mask_offset + h * Ht);
#endif

                uint32_t intermediates_idx = intermediates_offset + h * num_of_interm_tiles;
                read_tiles_by_row(
                    cb_intermediates,
                    intermediates_address_generator,
                    intermediates_idx,
                    num_of_interm_tiles,
                    interm_tile_bytes,
                    num_of_interm_tiles);

                const uint32_t vo_start_idx = vo_offset + (q_head_idx * Ht + h) * vWt;
                read_tiles_by_row(cb_grad_output, grad_output_address_generator, vo_start_idx, vWt, tile_bytes, vWt);

                read_one_tile(cb_u_scalar_row, u_scaler_address_generator, u_scaler_head_offset + h);
            }
            intermediates_offset += Ht * num_of_interm_tiles;
        }
    }
#endif
}
