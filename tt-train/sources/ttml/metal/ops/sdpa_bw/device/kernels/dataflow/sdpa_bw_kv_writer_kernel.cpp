// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compile_time_args.h>

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
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices for gradients
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_15;    // Output: grad_K
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_16;  // Output: grad_V

    // Get compile-time arguments
    constexpr uint32_t kWt = get_compile_time_arg_val(0);              // key width in tiles (grad_K)
    constexpr uint32_t vWt = get_compile_time_arg_val(1);              // value width in tiles (grad_V)
    constexpr uint32_t Ht = get_compile_time_arg_val(2);               // sequence length in tiles
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);          // number of query heads
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // heads per group
    // 5: chain chunk rows, 6: Q/K width in tiles, 7..11: chain semaphores (sender, receiver, valid, received,
    // forwarded)

#ifdef CAUSAL_MASK
    // Generate causal mask tile ONCE - will be reused for every diagonal
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
    generate_causal_mask_tile(cb_attn_mask);
#endif

    const uint32_t tile_bytes_k = get_tile_size(cb_grad_key);
    const uint32_t tile_bytes_v = get_tile_size(cb_grad_value);

    // TensorAccessor definitions with chained offsets
    constexpr auto grad_key_args = TensorAccessorArgs<12>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    // Create TensorAccessor generators for output gradients
    const auto grad_key_addr_generator = TensorAccessor(grad_key_args, grad_key_addr, tile_bytes_k);
    const auto grad_value_addr_generator = TensorAccessor(grad_value_args, grad_value_addr, tile_bytes_v);

    const uint32_t num_of_groups = q_heads / heads_per_group;

#if defined(WORK_LIST) || defined(CHAIN)
#ifdef HEAD_SPLIT
    constexpr uint32_t items_per_row = heads_per_group;
#else
    constexpr uint32_t items_per_row = 1U;
#endif
#ifdef CHAIN
    // Chain mode (iteration 23): this writer forwards the Q / lse / dO / u stream to the next core on its own
    // NoC while the reader receives the next chunk on the other (see the Q backward writer). After a round's
    // chunks the item's outputs go out as before.
    constexpr uint32_t chain_chunk = get_compile_time_arg_val(5);
    constexpr uint32_t qWt = get_compile_time_arg_val(6);
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_0;
    constexpr uint32_t cb_query = tt::CBIndex::c_2;
    constexpr uint32_t cb_intermediates = tt::CBIndex::c_6;
    constexpr uint32_t cb_u_scalar_row = tt::CBIndex::c_14;
    const uint32_t next_noc_x = num_rows_to_process;
    const uint32_t next_noc_y = start_row;
    const uint32_t is_sink = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t items_base = runtime_args_counter;
    const uint32_t starts_base =
        items_base + num_rounds;  // per round: the first query row the chain streams (iteration 27)
    Semaphore sender_sem(get_compile_time_arg_val(7));
    Semaphore receiver_sem(get_compile_time_arg_val(8));
    Semaphore received_sem(get_compile_time_arg_val(10));
    Semaphore forwarded_sem(get_compile_time_arg_val(11));
    const Noc noc;
    constexpr uint32_t kChainSlots = CHAIN_SLOTS;
    const uint32_t q_chunk_bytes = chain_chunk * qWt * get_tile_size(cb_query);
    const uint32_t interm_chunk_bytes = chain_chunk * get_tile_size(cb_intermediates);
    const uint32_t vo_chunk_bytes = chain_chunk * vWt * get_tile_size(cb_grad_output);
    const uint32_t u_chunk_bytes = chain_chunk * get_tile_size(cb_u_scalar_row);
    // kChainSlots chunks resident per CB; the reader's pushes are whole chunks, so chunk c sits in slot c %
    // kChainSlots.
    const uint32_t q_base = get_write_ptr(cb_query);
    const uint32_t i_base = get_write_ptr(cb_intermediates);
    const uint32_t vo_base = get_write_ptr(cb_grad_output);
    const uint32_t u_base = get_write_ptr(cb_u_scalar_row);
    uint32_t c = 0;  // chunk counter over the rounds (CB slot = c % kChainSlots)
    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t round_start = get_arg_val<uint32_t>(starts_base + round);
        for (uint32_t h = round_start; h < Ht; h += chain_chunk, ++c) {
            received_sem.wait_min(c + 1U);
            if (!is_sink) {
                const uint32_t slot = c % kChainSlots;
                sender_sem.wait_min(c + 1U);  // the next core's credits (iteration 26)
                const uint32_t q_addr = q_base + slot * q_chunk_bytes;
                const uint32_t i_addr = i_base + slot * interm_chunk_bytes;
                const uint32_t vo_addr = vo_base + slot * vo_chunk_bytes;
                const uint32_t u_addr = u_base + slot * u_chunk_bytes;
                noc.async_write(
                    CoreLocalMem<uint32_t>(q_addr),
                    UnicastEndpoint{},
                    q_chunk_bytes,
                    {},
                    {.noc_x = next_noc_x, .noc_y = next_noc_y, .addr = q_addr});
                noc.async_write(
                    CoreLocalMem<uint32_t>(i_addr),
                    UnicastEndpoint{},
                    interm_chunk_bytes,
                    {},
                    {.noc_x = next_noc_x, .noc_y = next_noc_y, .addr = i_addr});
                noc.async_write(
                    CoreLocalMem<uint32_t>(vo_addr),
                    UnicastEndpoint{},
                    vo_chunk_bytes,
                    {},
                    {.noc_x = next_noc_x, .noc_y = next_noc_y, .addr = vo_addr});
                noc.async_write(
                    CoreLocalMem<uint32_t>(u_addr),
                    UnicastEndpoint{},
                    u_chunk_bytes,
                    {},
                    {.noc_x = next_noc_x, .noc_y = next_noc_y, .addr = u_addr});
                noc.async_writes_flushed();
                receiver_sem.up(noc, next_noc_x, next_noc_y, 1);  // valid count
                forwarded_sem.set(c + 1U);
            }
        }
        const uint32_t item = get_arg_val<uint32_t>(items_base + round);
        if (item == 0xFFFFFFFFU) {  // a round without an item
            continue;
        }
        const uint32_t global_row_idx = item / items_per_row;
#else
    const uint32_t list_base = runtime_args_counter - 1U;
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t item = get_arg_val<uint32_t>(list_base + i);
        const uint32_t global_row_idx = item / items_per_row;
#endif
        const uint32_t batch_idx = global_row_idx / (num_of_groups * Ht);
        const uint32_t s_tile_idx = global_row_idx % Ht;
        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;
#ifdef HEAD_SPLIT
        // fp32 partial of query head (group, item % heads_per_group): (B, q_heads, S, D) layout.
        const uint32_t head_idx = item % items_per_row;
        const uint32_t out_row = (batch_idx * q_heads + group_idx * heads_per_group + head_idx) * Ht + s_tile_idx;
#else
        const uint32_t out_row = (batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx;
#endif
        write_tiles_by_row(cb_grad_value, grad_value_addr_generator, out_row * vWt, vWt, tile_bytes_v, vWt);
        write_tiles_by_row(cb_grad_key, grad_key_addr_generator, out_row * kWt, kWt, tile_bytes_k, kWt);
    }
#ifdef CHAIN
    noc.async_write_barrier();
    noc.async_atomic_barrier();
#endif
#elif defined(BALANCED_PARALLELISM)
    constexpr uint32_t pairs_per_seq = Ht / 2;

    auto write_row = [&](const uint32_t global_row_idx) {
        const uint32_t batch_idx = global_row_idx / (num_of_groups * Ht);
        const uint32_t s_tile_idx = global_row_idx % Ht;
        const uint32_t group_idx = (global_row_idx / Ht) % num_of_groups;

        const uint32_t grad_v_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * vWt;
        write_tiles_by_row(cb_grad_value, grad_value_addr_generator, grad_v_row_base_tiles, vWt, tile_bytes_v, vWt);

        const uint32_t grad_k_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * kWt;
        write_tiles_by_row(cb_grad_key, grad_key_addr_generator, grad_k_row_base_tiles, kWt, tile_bytes_k, kWt);
    };

    // Runtime args reuse: num_rows_to_process = num_pairs, start_row = start_pair_idx
    for (uint32_t p = 0; p < num_rows_to_process; ++p) {
        const uint32_t global_pair_idx = start_row + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        write_row(heavy_global_row);
        write_row(light_global_row);
    }
#else
    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; r++) {
        // Convert global row index to tensor coordinates
        const uint32_t batch_idx = r / (num_of_groups * Ht);
        const uint32_t s_tile_idx = r % Ht;  // position in sequence (tile idx)

        const uint32_t group_idx = (r / Ht) % num_of_groups;  // which group of K and V we are processing

        // -------- Grad Value: same shape as Value (B, vNH, S, vEmbd) --------
        const uint32_t grad_v_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * vWt;
        write_tiles_by_row(cb_grad_value, grad_value_addr_generator, grad_v_row_base_tiles, vWt, tile_bytes_v, vWt);

        // -------- Grad Key: same shape as Key (B, kNH, S, kEmbd) --------
        const uint32_t grad_k_row_base_tiles = ((batch_idx * num_of_groups + group_idx) * Ht + s_tile_idx) * kWt;
        write_tiles_by_row(cb_grad_key, grad_key_addr_generator, grad_k_row_base_tiles, kWt, tile_bytes_k, kWt);
    }
#endif
}
