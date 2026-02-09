// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <api/compile_time_args.h>
#include <api/debug/dprint.h>

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
    constexpr uint32_t cb_output = tt::CBIndex::c_15;

    constexpr uint32_t qWt = get_compile_time_arg_val(0);            // number of tiles in inner dimension
    constexpr uint32_t Ht = get_compile_time_arg_val(1);             // number of tiles in sequence dimension
    constexpr uint32_t q_heads = get_compile_time_arg_val(2);        // num of heads in query
    constexpr uint32_t pairs_per_seq = get_compile_time_arg_val(3);  // Ht/2 - pairs per sequence for balanced mode

    const uint32_t tile_bytes = get_tile_size(cb_output);

    constexpr auto output_args = TensorAccessorArgs<4>();
    const auto output_addr_generator = TensorAccessor(output_args, output_addr, tile_bytes);

#ifdef RETURN_INTERMEDIATES
    constexpr auto intermediates_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto intermediates_addr_generator = TensorAccessor(intermediates_args, intermediates_addr, tile_bytes);
#endif

    constexpr uint32_t onetile = 1U;

    // Generate tiles that compute kernel needs BEFORE reader starts pushing data
    // This allows reader to start DRAM reads immediately while writer generates these tiles
    constexpr uint32_t cb_reduction_scaler = tt::CBIndex::c_5;
    constexpr uint32_t cb_matmul_reduce = tt::CBIndex::c_6;

    constexpr uint16_t one = 0x00003F80;                          // (bfloat16)1.0 -> uint16_t
    generate_tile_with_bfloat16_value(cb_reduction_scaler, one);  // tile with 1.0 for reduction
    generate_matmul_row_reduce_tile(cb_matmul_reduce);            // tile for matmul row reduce

#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
    // Generate causal mask tile ONCE - will be reused for every diagonal
    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_3;
    generate_causal_mask_tile(cb_attn_mask);
#endif

    DPRINT << "WRITER: start=" << start_row << " rows=" << num_rows_to_process << ENDL();

    const uint32_t tiles_per_head = qWt;
    constexpr uint32_t kIntermediateTilesPerRow = 2U;

#ifdef BALANCED_PARALLELISM
    // Balanced parallelism mode: write outputs for pairs of rows (light + heavy)
    // Runtime args: num_rows_to_process = num_pairs, start_row = start_pair_idx
    const uint32_t num_pairs = num_rows_to_process;
    const uint32_t start_pair_idx = start_row;

    for (uint32_t p = 0; p < num_pairs; ++p) {
        const uint32_t global_pair_idx = start_pair_idx + p;

        // Map pair index to sequence and position within sequence
        const uint32_t seq_idx = global_pair_idx / pairs_per_seq;
        const uint32_t pair_in_seq = global_pair_idx % pairs_per_seq;

        // Calculate the two row indices for this pair
        const uint32_t light_row_in_seq = pair_in_seq;
        const uint32_t heavy_row_in_seq = Ht - 1 - pair_in_seq;

        const uint32_t light_global_row = seq_idx * Ht + light_row_in_seq;
        const uint32_t heavy_global_row = seq_idx * Ht + heavy_row_in_seq;

        // Write output for light row
        {
            const uint32_t r = light_global_row;
            const uint32_t s_tile_idx = r % Ht;
            const uint32_t q_head_idx = (r / Ht) % q_heads;
            const uint32_t batch_idx = r / (Ht * q_heads);
            const uint32_t out_start_idx = ((batch_idx * q_heads + q_head_idx) * Ht + s_tile_idx) * tiles_per_head;

            write_tiles_by_row(
                cb_output, output_addr_generator, out_start_idx, tiles_per_head, tile_bytes, tiles_per_head);

#ifdef RETURN_INTERMEDIATES
            const uint32_t intermediate_base_idx =
                ((batch_idx * q_heads + q_head_idx) * Ht + s_tile_idx) * kIntermediateTilesPerRow;

            cb_wait_front(cb_intermediates, kIntermediateTilesPerRow);
            uint32_t l1_intermediates_read_addr = get_read_ptr(cb_intermediates);
            noc_async_write_tile(intermediate_base_idx, intermediates_addr_generator, l1_intermediates_read_addr);
            l1_intermediates_read_addr += tile_bytes;
            noc_async_write_tile(intermediate_base_idx + 1, intermediates_addr_generator, l1_intermediates_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_intermediates, kIntermediateTilesPerRow);
#endif
        }

        // Write output for heavy row
        {
            const uint32_t r = heavy_global_row;
            const uint32_t s_tile_idx = r % Ht;
            const uint32_t q_head_idx = (r / Ht) % q_heads;
            const uint32_t batch_idx = r / (Ht * q_heads);
            const uint32_t out_start_idx = ((batch_idx * q_heads + q_head_idx) * Ht + s_tile_idx) * tiles_per_head;

            write_tiles_by_row(
                cb_output, output_addr_generator, out_start_idx, tiles_per_head, tile_bytes, tiles_per_head);

#ifdef RETURN_INTERMEDIATES
            const uint32_t intermediate_base_idx =
                ((batch_idx * q_heads + q_head_idx) * Ht + s_tile_idx) * kIntermediateTilesPerRow;

            cb_wait_front(cb_intermediates, kIntermediateTilesPerRow);
            uint32_t l1_intermediates_read_addr = get_read_ptr(cb_intermediates);
            noc_async_write_tile(intermediate_base_idx, intermediates_addr_generator, l1_intermediates_read_addr);
            l1_intermediates_read_addr += tile_bytes;
            noc_async_write_tile(intermediate_base_idx + 1, intermediates_addr_generator, l1_intermediates_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_intermediates, kIntermediateTilesPerRow);
#endif
        }
    }
    DPRINT << "WRITER DONE" << ENDL();
#else
    // Standard mode: write outputs sequentially
    const uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; r++) {
        // convert global row index to output tensor coordinates
        const uint32_t s_tile_idx = r % Ht;  // position in sequence (tile idx)
        const uint32_t q_head_idx = (r / Ht) % q_heads;
        const uint32_t batch_idx = r / (Ht * q_heads);

        // -------- Output: (B, H, S, D), heads NOT fused --------
        // Linear index for [B, H, S, D]: ((b * q_heads + h) * Ht + s_tile) * tiles_per_head + col
        const uint32_t out_start_idx = ((batch_idx * q_heads + q_head_idx) * Ht + s_tile_idx) * tiles_per_head;

        write_tiles_by_row(cb_output, output_addr_generator, out_start_idx, tiles_per_head, tile_bytes, tiles_per_head);

#ifdef RETURN_INTERMEDIATES
        // -------- Intermediates: (B, qNH, S, 64) = 2 tiles wide --------
        // Tile 0: max_val at col 0, rest padded
        // Tile 1: recip_sum_exp at col 32 (col 0 of second tile), rest padded
        // Linear index for [B, qNH, S, 64]: ((b * q_heads + h) * Ht + s_tile) * 2 + tile_offset
        const uint32_t intermediate_base_idx =
            ((batch_idx * q_heads + q_head_idx) * Ht + s_tile_idx) * kIntermediateTilesPerRow;

        cb_wait_front(cb_intermediates, kIntermediateTilesPerRow);
        uint32_t l1_intermediates_read_addr = get_read_ptr(cb_intermediates);

        // Write tile 0 (max_val)
        noc_async_write_tile(intermediate_base_idx, intermediates_addr_generator, l1_intermediates_read_addr);
        l1_intermediates_read_addr += tile_bytes;

        // Write tile 1 (recip_sum_exp)
        noc_async_write_tile(intermediate_base_idx + 1, intermediates_addr_generator, l1_intermediates_read_addr);

        noc_async_write_barrier();
        cb_pop_front(cb_intermediates, kIntermediateTilesPerRow);
#endif
    }
    DPRINT << "WRITER DONE" << ENDL();
#endif
}
