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
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t u_scaler_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t grad_gate_addr = get_arg_val<uint32_t>(runtime_args_counter++);          // 0 if no gate
    const uint32_t gated_grad_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);  // 0 if no gate
    const uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // Circular buffer indices
    constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_7;    // Matmul row reduce tile
    constexpr uint32_t cb_grad_query = tt::CBIndex::c_13;       // Output: grad_Q
    constexpr uint32_t cb_u_scaler_output = tt::CBIndex::c_14;  // Output: u_scaler for KV kernel
#ifdef HAS_GATE
    constexpr uint32_t cb_grad_gate = tt::CBIndex::c_16;          // Output: grad_gate = dY * Y * (1 - sigmoid(G))
    constexpr uint32_t cb_gated_grad_output = tt::CBIndex::c_17;  // Output: dO = dY * sigmoid(G) for KV kernel
#endif

    // Get compile-time arguments
    constexpr uint32_t qWt = get_compile_time_arg_val(0);  // query width in tiles
    constexpr uint32_t Ht = get_compile_time_arg_val(1);   // sequence length in tiles
    constexpr uint32_t vWt = get_compile_time_arg_val(2);  // V/dO width in tiles (grad_gate / gated dO rows)
    // 3: chain chunk rows, 4..8: chain semaphores (sender, receiver, valid, received, forwarded)

    // Generate helper tiles once at the start
    generate_matmul_row_reduce_tile(cb_mat_mul_reduce);  // tile for matmul row reduce

#ifdef CAUSAL_MASK
    // Generate causal mask tile ONCE - will be reused for every diagonal
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_5;
    generate_causal_mask_tile(cb_attn_mask);
#endif

    const uint32_t tile_bytes = get_tile_size(cb_grad_query);
    const uint32_t u_scaler_tile_bytes = get_tile_size(cb_u_scaler_output);

    // TensorAccessor definitions
    constexpr auto grad_query_args = TensorAccessorArgs<9>();
    constexpr auto u_scaler_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_gate_args = TensorAccessorArgs<u_scaler_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gated_grad_output_args =
        TensorAccessorArgs<grad_gate_args.next_compile_time_args_offset()>();

    // Create TensorAccessor generators
    const auto grad_query_addr_generator = TensorAccessor(grad_query_args, grad_query_addr);
    const auto u_scaler_addr_generator = TensorAccessor(u_scaler_args, u_scaler_output_addr);
#ifdef HAS_GATE
    const auto grad_gate_addr_generator = TensorAccessor(grad_gate_args, grad_gate_addr);
    const auto gated_grad_output_addr_generator = TensorAccessor(gated_grad_output_args, gated_grad_output_addr);
#endif

    auto write_u_scaler_for_row = [&](const uint32_t global_row_idx) {
        // u_scaler tensor: one tile per query row, indexed by global_row_idx
        cb_wait_front(cb_u_scaler_output, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_u_scaler_output);
        noc_async_write_page(global_row_idx, u_scaler_addr_generator, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_u_scaler_output, onetile);
    };

#ifdef HAS_GATE
    // grad_gate and gated dO both have grad_output's (B, H, S, vE) layout: one vWt-wide row per query row.
    auto write_gate_outputs_for_row = [&](const uint32_t global_row_idx) {
        const uint32_t vo_start_idx = global_row_idx * vWt;
        write_tiles_by_row(cb_grad_gate, grad_gate_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
        write_tiles_by_row(cb_gated_grad_output, gated_grad_output_addr_generator, vo_start_idx, vWt, tile_bytes, vWt);
    };
#endif

#ifdef CHAIN
    // Chain mode (iteration 23): this writer forwards the K/V stream to the next core on its own NoC while the
    // reader receives the next chunk on the other. Per chunk c (counted over the rounds): wait for the reader's
    // received = c + 1, wait for the next core's ready counter, write the chunk's K / V slots to the same CB
    // addresses there, flush, relay VALID, then publish forwarded = c + 1 (the reader waits for it before the
    // slot is reused). After a round's chunks the row's outputs go out as before.
    constexpr uint32_t chain_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t cb_key = tt::CBIndex::c_3;
    constexpr uint32_t cb_value = tt::CBIndex::c_4;
    const uint32_t next_noc_x = num_rows_to_process;
    const uint32_t next_noc_y = start_row;
    const uint32_t is_sink = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_rounds = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rows_base = runtime_args_counter;
    const uint32_t chunks_base = rows_base + num_rounds;  // per round: the chunks the chain streams (iteration 27)
    Semaphore sender_sem(get_compile_time_arg_val(4));
    Semaphore receiver_sem(get_compile_time_arg_val(5));
    Semaphore received_sem(get_compile_time_arg_val(7));
    Semaphore forwarded_sem(get_compile_time_arg_val(8));
    const Noc noc;
    constexpr uint32_t kChainSlots = CHAIN_SLOTS;
    constexpr uint32_t k_chunk_tiles = chain_chunk * qWt;
    constexpr uint32_t v_chunk_tiles = chain_chunk * vWt;
    const uint32_t kv_tile_bytes = get_tile_size(cb_key);
    const uint32_t k_chunk_bytes = k_chunk_tiles * kv_tile_bytes;
    const uint32_t v_chunk_bytes = v_chunk_tiles * kv_tile_bytes;
    // kChainSlots chunks resident per CB; the reader's pushes are whole chunks, so chunk c sits in slot c %
    // kChainSlots.
    const uint32_t k_base = get_write_ptr(cb_key);
    const uint32_t v_base = get_write_ptr(cb_value);
    uint32_t c = 0;  // chunk counter over the rounds (CB slot = c % kChainSlots)
    for (uint32_t round = 0; round < num_rounds; ++round) {
        const uint32_t round_chunks = get_arg_val<uint32_t>(chunks_base + round);
        for (uint32_t h = 0; h < round_chunks; ++h, ++c) {
            received_sem.wait_min(c + 1U);
            if (!is_sink) {
                const uint32_t k_addr = k_base + (c % kChainSlots) * k_chunk_bytes;
                const uint32_t v_addr = v_base + (c % kChainSlots) * v_chunk_bytes;
                sender_sem.wait_min(c + 1U);  // the next core's credits (iteration 26)
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
                receiver_sem.up(noc, next_noc_x, next_noc_y, 1);  // valid count
                forwarded_sem.set(c + 1U);
            }
        }
        const uint32_t r = get_arg_val<uint32_t>(rows_base + round);
        if (r == 0xFFFFFFFFU) {  // a round without a row
            continue;
        }
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, r * qWt, qWt, tile_bytes, qWt);
        write_u_scaler_for_row(r);
#ifdef HAS_GATE
        write_gate_outputs_for_row(r);
#endif
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
#elif defined(WORK_LIST)
    // Work-list mode: the runtime args from `start_row` on are this core's rows (heaviest first).
    const uint32_t list_base = runtime_args_counter - 1U;
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        const uint32_t r = get_arg_val<uint32_t>(list_base + i);
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, r * qWt, qWt, tile_bytes, qWt);
        write_u_scaler_for_row(r);
#ifdef HAS_GATE
        write_gate_outputs_for_row(r);
#endif
    }
#elif defined(BALANCED_PARALLELISM)
    constexpr uint32_t pairs_per_seq = Ht / 2;

    // Runtime args reuse: num_rows_to_process = num_pairs, start_row = start_pair_idx
    for (uint32_t p = 0; p < num_rows_to_process; ++p) {
        const uint32_t global_pair_idx = start_row + p;

        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        // Write heavy row grad_query + u_scaler (+ gate outputs)
        const uint32_t heavy_start_idx = heavy_global_row * qWt;
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, heavy_start_idx, qWt, tile_bytes, qWt);
        write_u_scaler_for_row(heavy_global_row);
#ifdef HAS_GATE
        write_gate_outputs_for_row(heavy_global_row);
#endif

        // Write light row grad_query + u_scaler (+ gate outputs)
        const uint32_t light_start_idx = light_global_row * qWt;
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, light_start_idx, qWt, tile_bytes, qWt);
        write_u_scaler_for_row(light_global_row);
#ifdef HAS_GATE
        write_gate_outputs_for_row(light_global_row);
#endif
    }
#else
    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        const uint32_t q_start_idx = r * qWt;

        // Write grad_query row
        write_tiles_by_row(cb_grad_query, grad_query_addr_generator, q_start_idx, qWt, tile_bytes, qWt);

        // Write u_scaler tile for this row
        write_u_scaler_for_row(r);

#ifdef HAS_GATE
        // Write grad_gate and gated dO rows
        write_gate_outputs_for_row(r);
#endif
    }
#endif
}
