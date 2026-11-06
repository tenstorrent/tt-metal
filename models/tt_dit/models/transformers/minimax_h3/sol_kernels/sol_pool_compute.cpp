// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, pool step, compute kernel.
//
// Per unit: out[dim_tile] = sum over tile rows r of pool_tile[r] @ data[r, dim_tile], accumulated in fp32 DST; units
// with no rows output zero tiles. Units and per-core ranges: sol_pool_units.hpp. With Q_OWN_CB, Q units read their data
// from cb_q_data (Q's dtype) and K/V units from cb_data.
//
// Formats: a matmul unpacks its first operand (the pooling tile) into srcB and its second (the data tile) into srcA,
// so the hardware is started in that order, and srcA is reconfigured whenever the next unit reads a CB of another
// format (bf8 K/V, bf16 Q, the bf16 zero tile). matmul_init does not reconfigure formats. The reconfigure is a no-op
// when the formats match, so all-bf16 runs do no extra work.
//
// Compile-time args: 0 NUM_HEADS, 1 SHARD_ROW_TILES, 2 HEAD_DIM_TILES, 3 SHARD_BLOCK_TILES, 4 SHARD_ROWS, 5 GRID_X,
// 6 GRID_Y, 7 NUM_UNITS, 8 Q_OWN_CB.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "sol_pool_units.hpp"

namespace {
constexpr uint32_t cb_data = 0, cb_q_data = 1, cb_pool_tiles = 2, cb_zero = 3, cb_out = 16;
constexpr uint32_t TILE_ROWS_PER_GROUP = sol_pool::TILE_ROWS_PER_GROUP;
constexpr uint32_t ROWS_PER_BATCH = 4;
inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(0);
    constexpr uint32_t SHARD_ROW_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t HEAD_DIM_TILES = get_compile_time_arg_val(2);
    constexpr uint32_t SHARD_BLOCK_TILES = get_compile_time_arg_val(3);
    static_assert(HEAD_DIM_TILES <= 4, "HEAD_DIM_TILES output tiles must fit the half-sync fp32 DST (4 tiles)");

    constexpr uint32_t GRID_X = get_compile_time_arg_val(5);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_UNITS = get_compile_time_arg_val(7);
    constexpr uint32_t Q_OWN_CB = get_compile_time_arg_val(8);
    constexpr uint32_t NUM_CORES = GRID_X * GRID_Y;
    using Layout = sol_pool::Layout<NUM_HEADS, SHARD_ROW_TILES, SHARD_BLOCK_TILES>;
    static_assert(Layout::NUM_UNITS == NUM_UNITS, "host unit count must match the kernel layout");
    const uint32_t core_index = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    uint32_t unit_start, unit_count;
    Layout::range(core_index, NUM_CORES, unit_start, unit_count);
    if (unit_count == 0) {
        return;
    }
    CircularBuffer pool_tiles_cb(cb_pool_tiles), zero_cb(cb_zero), out_cb(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_pool_tiles, cb_data, cb_out);  // srcB pooling, srcA data
    uint32_t cb_srca = cb_data;                                                    // the CB srcA is configured for
    zero_cb.wait_front(1);

    for (uint32_t unit = unit_start; unit < unit_start + unit_count; ++unit) {
        const sol_pool::Unit coords = Layout::decode(unit);
        const uint32_t group = coords.group;
        const uint32_t cb_unit_data = (Q_OWN_CB && coords.tensor == 0) ? cb_q_data : cb_data;
        CircularBuffer data_cb(cb_unit_data);
        const uint32_t first_row = group * TILE_ROWS_PER_GROUP;
        const uint32_t end_row = min_u(first_row + TILE_ROWS_PER_GROUP, SHARD_ROW_TILES);

        const uint32_t cb_next_srca = end_row <= first_row ? cb_zero : cb_unit_data;
        if (cb_next_srca != cb_srca) {
            reconfig_data_format_srca(cb_srca, cb_next_srca);
            cb_srca = cb_next_srca;
        }
        tile_regs_acquire();
        if (end_row <= first_row) {
            copy_init(cb_zero);
            for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                copy_tile(cb_zero, 0, dim_tile);
            }
        } else {
            matmul_init(cb_pool_tiles, cb_unit_data);
            for (uint32_t batch_row = first_row; batch_row < end_row; batch_row += ROWS_PER_BATCH) {
                const uint32_t rows_in_batch = min_u(ROWS_PER_BATCH, end_row - batch_row);
                data_cb.wait_front(ROWS_PER_BATCH * HEAD_DIM_TILES);  // batches are always full
                pool_tiles_cb.wait_front(ROWS_PER_BATCH);
                for (uint32_t row = 0; row < rows_in_batch; ++row) {
                    for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
                        matmul_tiles(cb_pool_tiles, cb_unit_data, row, row * HEAD_DIM_TILES + dim_tile, dim_tile);
                    }
                }
                pool_tiles_cb.pop_front(ROWS_PER_BATCH);
                data_cb.pop_front(ROWS_PER_BATCH * HEAD_DIM_TILES);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        out_cb.reserve_back(HEAD_DIM_TILES);
        for (uint32_t dim_tile = 0; dim_tile < HEAD_DIM_TILES; ++dim_tile) {
            pack_tile<true>(dim_tile, cb_out, dim_tile);
        }
        out_cb.push_back(HEAD_DIM_TILES);
        tile_regs_release();
    }
}
