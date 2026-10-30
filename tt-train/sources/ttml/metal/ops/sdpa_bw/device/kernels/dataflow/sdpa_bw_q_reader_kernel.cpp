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
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);    // grad_output buffer address
    const uint32_t attn_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);    // attn_output buffer address
    const uint32_t query_addr = get_arg_val<uint32_t>(runtime_args_counter++);          // query buffer address
    const uint32_t key_addr = get_arg_val<uint32_t>(runtime_args_counter++);            // key buffer address
    const uint32_t value_addr = get_arg_val<uint32_t>(runtime_args_counter++);          // value buffer address
    const uint32_t mask_addr = get_arg_val<uint32_t>(runtime_args_counter++);           // mask buffer address
    const uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);  // intermediates buffer address
    const uint32_t gate_addr = get_arg_val<uint32_t>(runtime_args_counter++);  // gate buffer address (0 if unused)
    const uint32_t num_rows_to_process =
        get_arg_val<uint32_t>(runtime_args_counter++);                         // rows to process in this kernel
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);  // starting row for this core

    // Circular buffer indices
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;
    constexpr uint32_t cb_attn_output = tt::CBIndex::c_1;
    constexpr uint32_t cb_query = tt::CBIndex::c_2;
    constexpr uint32_t cb_key = tt::CBIndex::c_3;
    constexpr uint32_t cb_value = tt::CBIndex::c_4;
#ifdef USE_ATTN_MASK
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
#endif
    constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;
#ifdef HAS_GATE
    constexpr uint32_t cb_gate = tt::CBIndex::c_15;
#endif

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);              // Q/K width in tiles
    constexpr uint32_t vWt = get_compile_time_arg_val(1);              // V/dO/O width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(2);               // sequence length in tiles
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // heads per group

    // TensorAccessor definitions with chained offsets
    // compile-time args 5..8: chain-mode semaphores and chunk rows
    constexpr auto grad_output_args = TensorAccessorArgs<9>();
    constexpr auto attn_output_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto query_args = TensorAccessorArgs<attn_output_args.next_compile_time_args_offset()>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto intermediates_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
#ifdef HAS_GATE
    constexpr auto gate_args = TensorAccessorArgs<intermediates_args.next_compile_time_args_offset()>();
#endif

    const uint32_t tile_bytes = get_tile_size(cb_grad_output);

    // Create TensorAccessor generators for inputs
    const auto grad_output_addr_generator = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto attn_output_addr_generator = TensorAccessor(attn_output_args, attn_output_addr, tile_bytes);
    const auto query_addr_generator = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key_addr_generator = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value_addr_generator = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto mask_addr_generator = TensorAccessor(mask_args, mask_addr, tile_bytes);

    const uint32_t interm_tile_bytes = get_tile_size(cb_intermediates);
    const auto intermediates_addr_generator = TensorAccessor(intermediates_args, intermediates_addr, interm_tile_bytes);
#ifdef HAS_GATE
    const auto gate_addr_generator = TensorAccessor(gate_args, gate_addr, tile_bytes);
#endif

    const uint32_t num_of_groups = q_heads / heads_per_group;
    const uint32_t num_of_interm_tiles = 1U;

#if defined(BALANCED_PARALLELISM) || defined(WORK_LIST)
    constexpr uint32_t pairs_per_seq = Ht / 2;

    auto read_row = [&](const uint32_t global_row_idx) {
        const uint32_t vo_start_idx = global_row_idx * vWt;
        read_tiles_by_row(cb_attn_output, attn_output_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
        read_tiles_by_row(cb_grad_output, grad_output_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);

        const uint32_t q_start_idx = global_row_idx * qWt;
        read_tiles_by_row(cb_query, query_addr_generator, q_start_idx, qWt, tile_bytes, qWt);

#ifdef HAS_GATE
        // Gate row has the same (B, H, S, vE) layout as grad_output, so it shares vo_start_idx
        read_tiles_by_row(cb_gate, gate_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
#endif

        const uint32_t q_head_idx = (global_row_idx / Ht) % q_heads;
        const uint32_t batch_idx = global_row_idx / (Ht * q_heads);
        const uint32_t kv_group_idx = q_head_idx / heads_per_group;
        const uint32_t key_offset = (batch_idx * num_of_groups + kv_group_idx) * Ht * qWt;
        const uint32_t value_offset = (batch_idx * num_of_groups + kv_group_idx) * Ht * vWt;
        const uint32_t q_row_tile = global_row_idx % Ht;
        const uint32_t num_kv_tiles_to_read = q_row_tile + 1;

        const uint32_t intermediates_idx = global_row_idx * num_of_interm_tiles;
        read_tiles_by_row(
            cb_intermediates,
            intermediates_addr_generator,
            intermediates_idx,
            num_of_interm_tiles,
            interm_tile_bytes,
            num_of_interm_tiles);

        for (uint32_t h = 0; h < num_kv_tiles_to_read; ++h) {
            read_tiles_by_row(cb_key, key_addr_generator, key_offset + h * qWt, qWt, tile_bytes, qWt);
            read_tiles_by_row(cb_value, value_addr_generator, value_offset + h * vWt, vWt, tile_bytes, vWt);
        }
    };

#ifdef CHAIN
    // Chain mode: per round the row's own operands (if any), then the stream's K/V rows in chunks of
    // chain_chunk rows: the injector reads a chunk from DRAM, the other cores receive it from the previous
    // core; every core but the sink forwards it to the next core (ready counter, data, valid flag).
    constexpr uint32_t chain_chunk = get_compile_time_arg_val(8);
    const uint32_t prev_noc_x = num_rows_to_process;
    const uint32_t prev_noc_y = start_row;
    const uint32_t next_noc_x = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t next_noc_y = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t is_injector = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t is_sink = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t kv_stream = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;

    Semaphore sender_sem(get_compile_time_arg_val(5));
    Semaphore receiver_sem(get_compile_time_arg_val(6));
    Semaphore valid_sem(get_compile_time_arg_val(7));
    const Noc noc;
    valid_sem.set(VALID);

    const uint32_t key_offset = kv_stream * Ht * qWt;
    const uint32_t value_offset = kv_stream * Ht * vWt;
    constexpr uint32_t k_chunk_tiles = chain_chunk * qWt;
    constexpr uint32_t v_chunk_tiles = chain_chunk * vWt;
    const uint32_t k_chunk_bytes = k_chunk_tiles * tile_bytes;
    const uint32_t v_chunk_bytes = v_chunk_tiles * tile_bytes;

    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t global_row_idx = get_arg_val<uint32_t>(rows_base + round);
        if (global_row_idx != 0xFFFFFFFFU) {
            const uint32_t vo_start_idx = global_row_idx * vWt;
            read_tiles_by_row(cb_attn_output, attn_output_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
            read_tiles_by_row(cb_grad_output, grad_output_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
            read_tiles_by_row(cb_query, query_addr_generator, global_row_idx * qWt, qWt, tile_bytes, qWt);
#ifdef HAS_GATE
            read_tiles_by_row(cb_gate, gate_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
#endif
            read_tiles_by_row(
                cb_intermediates,
                intermediates_addr_generator,
                global_row_idx * num_of_interm_tiles,
                num_of_interm_tiles,
                interm_tile_bytes,
                num_of_interm_tiles);
        }
        for (uint32_t h0 = 0; h0 < Ht; h0 += chain_chunk) {
            cb_reserve_back(cb_key, k_chunk_tiles);
            cb_reserve_back(cb_value, v_chunk_tiles);
            const uint32_t k_addr = get_write_ptr(cb_key);
            const uint32_t v_addr = get_write_ptr(cb_value);
            if (is_injector) {
                for (uint32_t h = h0; h < h0 + chain_chunk; ++h) {
                    read_tiles_by_row(cb_key, key_addr_generator, key_offset + h * qWt, qWt, tile_bytes, qWt);
                    read_tiles_by_row(cb_value, value_addr_generator, value_offset + h * vWt, vWt, tile_bytes, vWt);
                }
            } else {
                receiver_sem.set(INVALID);
                sender_sem.up(noc, prev_noc_x, prev_noc_y, 1);
                receiver_sem.wait(VALID);
                cb_push_back(cb_key, k_chunk_tiles);
                cb_push_back(cb_value, v_chunk_tiles);
            }
            if (!is_sink) {
                sender_sem.wait(1);
                sender_sem.set(0);
                noc.async_write(
                    CoreLocalMem<uint32_t>(k_addr),
                    UnicastEndpoint{},
                    k_chunk_bytes,
                    {},
                    {.noc_x = next_noc_x, .noc_y = next_noc_y, .addr = k_addr});
                noc.async_write(
                    CoreLocalMem<uint32_t>(v_addr),
                    UnicastEndpoint{},
                    v_chunk_bytes,
                    {},
                    {.noc_x = next_noc_x, .noc_y = next_noc_y, .addr = v_addr});
                noc.async_writes_flushed();
                valid_sem.relay_unicast(noc, receiver_sem, next_noc_x, next_noc_y);
            }
        }
    }
    noc.async_atomic_barrier();
#elif defined(WORK_LIST)
    // Work-list mode: the runtime args from `start_row` on are this core's rows (heaviest first).
    const uint32_t list_base = runtime_args_counter - 1U;
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        read_row(get_arg_val<uint32_t>(list_base + i));
    }
#else
    // Runtime args reuse: num_rows_to_process = num_pairs, start_row = start_pair_idx
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
#endif
#else
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t global_row_idx = start_row + i;

        // Read attn_output and grad_output rows (V-dim width)
        const uint32_t vo_start_idx = global_row_idx * vWt;
        read_tiles_by_row(cb_attn_output, attn_output_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
        read_tiles_by_row(cb_grad_output, grad_output_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);

        // Read query row (Q/K-dim width)
        const uint32_t q_start_idx = global_row_idx * qWt;
        read_tiles_by_row(cb_query, query_addr_generator, q_start_idx, qWt, tile_bytes, qWt);

#ifdef HAS_GATE
        // Read gate row (V-dim width)
        read_tiles_by_row(cb_gate, gate_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
#endif

        const uint32_t q_head_idx = (global_row_idx / Ht) % q_heads;  // which head of Q we are processing right now

        // which batch we are processing right now
        const uint32_t batch_idx = global_row_idx / (Ht * q_heads);
        // calculate which group of K and V we need to read for this head of Q
        const uint32_t kv_group_idx = q_head_idx / heads_per_group;

        // calculate the starting index of K and V to read (separate offsets due to different inner dims)
        const uint32_t key_offset = (batch_idx * num_of_groups + kv_group_idx) * Ht * qWt;
        const uint32_t value_offset = (batch_idx * num_of_groups + kv_group_idx) * Ht * vWt;

        // q_row_tile = position within sequence (0 to Ht-1)
        const uint32_t q_row_tile = global_row_idx % Ht;

#ifdef CAUSAL_MASK
        // For causal mask: only read K/V tiles up to and including the diagonal
        const uint32_t num_kv_tiles_to_read = q_row_tile + 1;
#else
        const uint32_t num_kv_tiles_to_read = Ht;
#endif

#ifdef USE_ATTN_MASK
        // Mask is (1, 1, S, S) - same mask for all batches/heads, indexed by sequence position only
        const uint32_t mask_offset = q_row_tile * Ht;
#endif

        // read intermediates for current row of Q
        // intermediates shape: (B, qNH, S, 32) -> (batch, heads, seq_len, 1 FP32 tile)
        const uint32_t intermediates_idx = global_row_idx * num_of_interm_tiles;
        read_tiles_by_row(
            cb_intermediates,
            intermediates_addr_generator,
            intermediates_idx,
            num_of_interm_tiles,
            interm_tile_bytes,
            num_of_interm_tiles);

        for (uint32_t h = 0; h < num_kv_tiles_to_read; ++h) {
            // Read one row of K (Q/K-dim width)
            read_tiles_by_row(cb_key, key_addr_generator, key_offset + h * qWt, qWt, tile_bytes, qWt);

#ifdef USE_ATTN_MASK
            // Read one tile of attn_mask for current row of K and V
            // Row of K defines the column in (QK^T) matrix, so it defines the column of attn_mask to read
            read_one_tile(cb_attn_mask, mask_addr_generator, mask_offset + h);
#endif
            // Note: For CAUSAL_MASK, the mask tile is generated once by writer and reused by compute

            // Read one row of V (V-dim width)
            read_tiles_by_row(cb_value, value_addr_generator, value_offset + h * vWt, vWt, tile_bytes, vWt);
        }
    }
#endif
}
