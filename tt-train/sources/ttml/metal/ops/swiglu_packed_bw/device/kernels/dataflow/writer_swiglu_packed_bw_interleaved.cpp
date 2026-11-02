// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/common/swiglu_packed_layout.hpp"

constexpr uint32_t cb_dgate_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_dup_idx = tt::CBIndex::c_4;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t blocks_per_row = get_compile_time_arg_val(1);
constexpr uint32_t row_stride = get_compile_time_arg_val(2);      // tiles between consecutive rows of a half
constexpr uint32_t up_offset = get_compile_time_arg_val(3);       // tiles from a dgate tile to its dup tile
constexpr uint32_t col_major_rows = get_compile_time_arg_val(4);  // column-major block order: rows per column

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    const uint32_t dpacked_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dgate_idx);
    constexpr auto dpacked_args = TensorAccessorArgs<5>();
    const auto dpacked_gen = TensorAccessor(dpacked_args, dpacked_addr);

    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
#if defined(TRANSPOSED_GRAD) || defined(TRANSPOSED_IO)
        // column-major blocks (iteration 38): block b is column block b / col_major_rows of row b % col_major_rows
        const uint32_t dgate_start = (b % col_major_rows) * row_stride + (b / col_major_rows) * block_size;
#else
        const uint32_t dgate_start = swiglu_packed_block_tile(b, blocks_per_row, row_stride, block_size);
#endif
        // Disjoint tiles of the same dpacked buffer.
        write_tiles_by_row(cb_dgate_idx, dpacked_gen, dgate_start, block_size, tile_bytes, block_size);
        write_tiles_by_row(cb_dup_idx, dpacked_gen, dgate_start + up_offset, block_size, tile_bytes, block_size);
    }
}
