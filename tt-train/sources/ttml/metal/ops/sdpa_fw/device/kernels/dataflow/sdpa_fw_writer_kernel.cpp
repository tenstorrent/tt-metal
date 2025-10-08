// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <compile_time_args.h>
#include <debug/dprint.h>

#include <cstdint>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t intermediates_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_intermediates = tt::CBIndex::c_4;
    constexpr uint32_t cb_output = tt::CBIndex::c_15;

    constexpr uint32_t qWt = get_compile_time_arg_val(0);  // number of tiles in inner dimension
    constexpr uint32_t Ht = get_compile_time_arg_val(1);   // number of tiles in sequence dimension
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t q_heads = get_compile_time_arg_val(3);          // num of heads in query
    constexpr uint32_t heads_per_group = get_compile_time_arg_val(4);  // num of heads per group

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const DataFormat data_format = get_dataformat(cb_output);

    constexpr auto output_args = TensorAccessorArgs<5>();
    const auto output_addr_generator = TensorAccessor(output_args, output_addr, tile_bytes);

#ifdef RETURN_INTERMEDIATES
    constexpr auto intermediates_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto intermediates_addr_generator = TensorAccessor(intermediates_args, intermediates_addr, tile_bytes);
#endif

    constexpr uint32_t onetile = 1U;

    const uint32_t tiles_per_head = qWt;
    const uint32_t outWt = tiles_per_head * q_heads;  // fused width in tiles: (qNH * d) / TILE_W

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; r++) {
        // convert global row index to output tensor coordinates
        uint32_t s_tile_idx = r % Ht;  // position in sequence (tile idx)
        uint32_t q_head_idx = (r / Ht) % q_heads;
        uint32_t batch_idx = r / (Ht * q_heads);

        // -------- Output: (B, 1, S, qNH*qEmbd), heads fused in last dim --------
        // Row base for (batch_idx, s_tile): ((b * 1 + 0) * Ht + s_tile_idx) * outWt
        uint32_t out_row_base_tiles = ((batch_idx * Ht) + s_tile_idx) * outWt;

        // Slice for this head in fused width
        uint32_t head_offset_tiles = q_head_idx * tiles_per_head;

        // First tile index where we place this head's row
        uint32_t out_start_idx = out_row_base_tiles + head_offset_tiles;

        cb_wait_front(cb_output, tiles_per_head);
        uint32_t l1_read_addr = get_read_ptr(cb_output);
        for (uint32_t col = 0; col < tiles_per_head; ++col) {
            noc_async_write_tile(out_start_idx + col, output_addr_generator, l1_read_addr);
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, tiles_per_head);

#ifdef RETURN_INTERMEDIATES
        // -------- Intermediates: (B, qNH, S, 1U) --------
        // One tile per (b, h, s). Reduced value already packed in column 0, rest padded.
        // Linear index for [B, qNH, S, 1]: ((b * q_heads + h) * Ht + s_tile)
        uint32_t intermediate_idx = ((batch_idx * q_heads + q_head_idx) * Ht) + s_tile_idx;

        cb_wait_front(cb_intermediates, onetile);
        uint32_t l1_intermediates_read_addr = get_read_ptr(cb_intermediates);
        noc_async_write_tile(intermediate_idx, intermediates_addr_generator, l1_intermediates_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_intermediates, onetile);
#endif
    }
}
