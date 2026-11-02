// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per output tile (branch, column): the block tile `column`, the residual tile hidden_tiles * branch + column and the
// injection tile with its column `branch` copied into column 0 (rows 0..31), which the compute column-broadcasts.
// Named compile-time args: cb_block, cb_res, cb_coef, hidden_tiles, branches.  Compile-time args: TensorAccessorArgs
// for block, residual, injection, chained from 0.
// Runtime args: 0 block addr, 1 residual addr, 2 injection addr, 3 first unit, 4 units; unit u = (column u / branches,
// branch u % branches).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t block_addr = get_arg_val<uint32_t>(0);
    const uint32_t residual_addr = get_arg_val<uint32_t>(1);
    const uint32_t injection_addr = get_arg_val<uint32_t>(2);
    const uint32_t first_unit = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_block = get_named_compile_time_arg_val("cb_block");
    constexpr uint32_t cb_res = get_named_compile_time_arg_val("cb_res");
    constexpr uint32_t cb_coef = get_named_compile_time_arg_val("cb_coef");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t branches = get_named_compile_time_arg_val("branches");
    constexpr auto block_args = TensorAccessorArgs<0>();
    constexpr auto residual_args = TensorAccessorArgs<block_args.next_compile_time_args_offset()>();
    constexpr auto injection_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();

    const auto block = TensorAccessor(block_args, block_addr);
    const auto residual = TensorAccessor(residual_args, residual_addr);
    const auto injection = TensorAccessor(injection_args, injection_addr);
    const uint32_t tile_bytes = get_tile_size(cb_block);
    const Noc noc;
    DataflowBuffer block_tile(cb_block);
    DataflowBuffer residual_tile(cb_res);
    DataflowBuffer injection_tile(cb_coef);

    for (uint32_t u = first_unit; u < first_unit + units; ++u) {
        const uint32_t branch = u % branches;
        const uint32_t column = u / branches;
        block_tile.reserve_back(1);
        residual_tile.reserve_back(1);
        injection_tile.reserve_back(1);
        noc.async_read(block, block_tile, tile_bytes, {.page_id = column}, {.offset_bytes = 0});
        noc.async_read(
            residual, residual_tile, tile_bytes, {.page_id = hidden_tiles * branch + column}, {.offset_bytes = 0});
        noc.async_read(injection, injection_tile, tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        // bf16 tile element (r, c) is 16-bit word (r >> 4) * 512 + (c >> 4) * 256 + (r & 15) * 16 + (c & 15)
        const CoreLocalMem<volatile uint16_t> injection_words(injection_tile.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t row_start = (r >> 4) * 512 + (r & 15) * 16;
            injection_words[row_start] = injection_words[row_start + branch];
        }
        block_tile.push_back(1);
        residual_tile.push_back(1);
        injection_tile.push_back(1);
    }
}
