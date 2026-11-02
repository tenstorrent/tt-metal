// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/common/swiglu_packed_layout.hpp"

constexpr uint32_t cb_gate_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_up_idx = tt::CBIndex::c_1;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t blocks_per_row = get_compile_time_arg_val(1);
constexpr uint32_t row_stride = get_compile_time_arg_val(2);      // tiles between consecutive rows of a half
constexpr uint32_t up_offset = get_compile_time_arg_val(3);       // tiles from a gate tile to its up tile
constexpr uint32_t col_major_rows = get_compile_time_arg_val(4);  // COL_MAJOR: rows per column block

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t packed_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t start_block = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_gate_idx);
    constexpr auto packed_args = TensorAccessorArgs<5>();
    const auto packed_gen = TensorAccessor(packed_args, packed_addr);

    const uint32_t end_block = start_block + num_blocks_to_process;
    for (uint32_t b = start_block; b < end_block; ++b) {
#ifdef COL_MAJOR
        // Column-major blocks (iteration 39, the transposed layout): a core's consecutive blocks step down the
        // I-rows at one token column block, so its DRAM pages go in strides of a row (the faster pattern,
        // `O/iter39_swiglu_time.log`).
        const uint32_t gate_start = (b % col_major_rows) * row_stride + (b / col_major_rows) * block_size;
#else
        const uint32_t gate_start = swiglu_packed_block_tile(b, blocks_per_row, row_stride, block_size);
#endif

        read_tiles_by_row<false>(cb_gate_idx, packed_gen, gate_start, block_size, tile_bytes, block_size);
        read_tiles_by_row<false>(cb_up_idx, packed_gen, gate_start + up_offset, block_size, tile_bytes, block_size);
        noc_async_read_barrier();
        cb_push_back(cb_gate_idx, block_size);
        cb_push_back(cb_up_idx, block_size);
    }
}
