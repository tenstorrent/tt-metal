// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, pool step, writer kernel: stores each unit's HEAD_DIM_TILES output tiles at (head, tile row
// group) of the output tensor [1, NUM_HEADS, SHARD_BLOCKS_PADDED, head_dim].
//
// Compile-time args: 0 NUM_HEADS, 1 SHARD_ROW_TILES, 2 HEAD_DIM_TILES, 3 SHARD_BLOCK_TILES, 4 SHARD_ROWS, 5 GRID_X,
// 6 GRID_Y, 7 NUM_UNITS, 8 Q_OWN_CB (unused here), then TensorAccessorArgs of the Q, K, V outputs.
// Common runtime args: 0..2 the Q, K, V output addresses. Units and per-core ranges: sol_pool_units.hpp.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "sol_pool_units.hpp"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t TILE_BYTES = 2048;
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(0);
    constexpr uint32_t SHARD_ROW_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t HEAD_DIM_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t SHARD_BLOCK_TILES = get_compile_time_arg_val(3);
    constexpr auto q_out_args = TensorAccessorArgs<9>();
    constexpr auto k_out_args = TensorAccessorArgs<q_out_args.next_compile_time_args_offset()>();
    constexpr auto v_out_args = TensorAccessorArgs<k_out_args.next_compile_time_args_offset()>();

    constexpr uint32_t GRID_X = get_compile_time_arg_val(5);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(7);
    constexpr uint32_t NUM_CORES = GRID_X * GRID_Y;
    using Layout = sol_pool::Layout<NUM_HEADS, SHARD_ROW_TILES, SHARD_BLOCK_TILES>;
    static_assert(Layout::NUM_UNITS == NUM_UNITS, "host unit count must match the kernel layout");
    const uint32_t core_index = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    uint32_t unit_start, unit_count;
    Layout::range(core_index, NUM_CORES, unit_start, unit_count);
    if (unit_count == 0) {
        return;
    }
    const auto q_out = TensorAccessor(q_out_args, get_common_arg_val<uint32_t>(0));
    const auto k_out = TensorAccessor(k_out_args, get_common_arg_val<uint32_t>(1));
    const auto v_out = TensorAccessor(v_out_args, get_common_arg_val<uint32_t>(2));

    Noc noc;
    CircularBuffer out_cb(cb_out);
    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        const sol_pool::Unit coords = Layout::decode(unit);
        out_cb.wait_front(HEAD_DIM_TILES);
        auto store = [&](auto& accessor) {
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                noc.async_write(
                    out_cb,
                    accessor,
                    TILE_BYTES,
                    {.offset_bytes = dim_tile * TILE_BYTES},
                    {.page_id = (coords.head * SHARD_BLOCK_TILES + coords.group) * HEAD_DIM_TILES + dim_tile});
            }
        };
        if (coords.tensor == 0) {
            store(q_out);
        } else if (coords.tensor == 1) {
            store(k_out);
        } else {
            store(v_out);
        }
        noc.async_write_barrier();
        out_cb.pop_front(HEAD_DIM_TILES);
    }
}
