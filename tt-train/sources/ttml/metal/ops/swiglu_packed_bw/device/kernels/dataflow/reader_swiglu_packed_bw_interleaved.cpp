// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/common/swiglu_packed_layout.hpp"

constexpr uint32_t cb_gate_idx = tt::CBIndex::c_0;  // gate branch (silu'd), from packed[:, :I]
constexpr uint32_t cb_up_idx = tt::CBIndex::c_1;    // up branch, from packed[:, I:]
constexpr uint32_t cb_dh_idx = tt::CBIndex::c_2;    // upstream grad dL/dh

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t blocks_per_row = get_compile_time_arg_val(1);
constexpr uint32_t row_stride = get_compile_time_arg_val(2);      // tiles between consecutive rows of a half
constexpr uint32_t up_offset = get_compile_time_arg_val(3);       // tiles from a gate tile to its up tile
constexpr uint32_t Wt_dh = get_compile_time_arg_val(4);           // dL/dh row stride in tiles (>= Wt); Ht transposed
constexpr uint32_t col_major_rows = get_compile_time_arg_val(5);  // column-major block order: rows per column
constexpr uint32_t tail_start = get_compile_time_arg_val(6);      // first tile of dh^T's LoRA tail row block
#if defined(TRANSPOSED_GRAD) || defined(TRANSPOSED_IO)
constexpr uint32_t cb_dh_t_idx = tt::CBIndex::c_8;   // dh^T tiles as read
constexpr uint32_t cb_a_idx = tt::CBIndex::c_9;      // lora_a tiles
constexpr uint32_t cb_tail_idx = tt::CBIndex::c_10;  // s_dh^T tiles
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t packed_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dh_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    [[maybe_unused]] const uint32_t a_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_gate_idx);
    constexpr auto packed_args = TensorAccessorArgs<7>();
    constexpr auto dh_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
    const auto packed_gen = TensorAccessor(packed_args, packed_addr);
    const auto dh_gen = TensorAccessor(dh_args, dh_addr);
#if defined(TRANSPOSED_GRAD) || defined(TRANSPOSED_IO)
    constexpr auto a_args = TensorAccessorArgs<dh_args.next_compile_time_args_offset()>();
    const auto a_gen = TensorAccessor(a_args, a_addr);
#endif

    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
#if defined(TRANSPOSED_GRAD) || defined(TRANSPOSED_IO)
        // Column-major blocks (iteration 38): block b is column block b / col_major_rows of row b % col_major_rows,
        // so a core's consecutive blocks share what belongs to the column block (TRANSPOSED_GRAD: the lora_a tiles;
        // TRANSPOSED_IO: the s_dh^T tiles), read once per column block (new_col).
        const uint32_t row = b % col_major_rows;
        const uint32_t col = (b / col_major_rows) * block_size;
        const uint32_t gate_start = row * row_stride + col;
        const bool new_col = (b == start_block) || (row == 0);
#else
        const uint32_t gate_start = swiglu_packed_block_tile(b, blocks_per_row, row_stride, block_size);
        const uint32_t row = b / blocks_per_row;
        const uint32_t col = (b - row * blocks_per_row) * block_size;
#endif

        read_tiles_by_row<false>(cb_gate_idx, packed_gen, gate_start, block_size, tile_bytes, block_size);
        read_tiles_by_row<false>(cb_up_idx, packed_gen, gate_start + up_offset, block_size, tile_bytes, block_size);
#ifdef TRANSPOSED_GRAD
        // Token row `row`, columns [col, col + block_size) of dh^T (pages (col + j) * Ht + row), the lora_a tiles of
        // those columns once per column block, the row's s_dh^T tile (tile row Wt of dh^T) every block.
        cb_reserve_back(cb_dh_t_idx, block_size);
        cb_reserve_back(cb_tail_idx, 1);
        if (new_col) {
            cb_reserve_back(cb_a_idx, block_size);
        }
        {
            uint32_t l1_dh = get_write_ptr(cb_dh_t_idx);
            uint32_t l1_a = get_write_ptr(cb_a_idx);
            for (uint32_t j = 0; j < block_size; ++j) {
                noc_async_read_page((col + j) * Wt_dh + row, dh_gen, l1_dh);
                l1_dh += tile_bytes;
                if (new_col) {
                    noc_async_read_page(col + j, a_gen, l1_a);
                    l1_a += tile_bytes;
                }
            }
            noc_async_read_page(tail_start + row, dh_gen, get_write_ptr(cb_tail_idx));
        }
#elif defined(TRANSPOSED_IO)
        // I-row `row`, token columns [col, col + block_size) of dh^T wide (pages row * Ht + col + j, a plain row
        // read), the row's lora_a tile (tile column `row` of lora_a [32, I]) every block, the column block's s_dh^T
        // tiles (tail_start + col + j) once per column block (iteration 39).
        read_tiles_by_row<false>(cb_dh_t_idx, dh_gen, row * Wt_dh + col, block_size, tile_bytes, block_size);
        cb_reserve_back(cb_a_idx, 1);
        noc_async_read_page(row, a_gen, get_write_ptr(cb_a_idx));
        if (new_col) {
            read_tiles_by_row<false>(cb_tail_idx, dh_gen, tail_start + col, block_size, tile_bytes, block_size);
        }
#else
        read_tiles_by_row<false>(cb_dh_idx, dh_gen, row * Wt_dh + col, block_size, tile_bytes, block_size);
#endif
        noc_async_read_barrier();
        cb_push_back(cb_gate_idx, block_size);
        cb_push_back(cb_up_idx, block_size);
#ifdef TRANSPOSED_GRAD
        cb_push_back(cb_dh_t_idx, block_size);
        cb_push_back(cb_tail_idx, 1);
        if (new_col) {
            cb_push_back(cb_a_idx, block_size);
        }
#elif defined(TRANSPOSED_IO)
        cb_push_back(cb_dh_t_idx, block_size);
        cb_push_back(cb_a_idx, 1);
        if (new_col) {
            cb_push_back(cb_tail_idx, block_size);
        }
#else
        cb_push_back(cb_dh_idx, block_size);
#endif
    }
}
