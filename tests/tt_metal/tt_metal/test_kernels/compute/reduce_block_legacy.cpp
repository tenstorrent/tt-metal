// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) REDUCE_SCALAR SUM block kernel, classic circular buffers: c_0 = data, c_1 = scaler,
// c_16 = reduced output. Reduces a block of num_tiles data tiles (each with scaler tile 0) accumulating into
// DST[0] by looping the legacy reduce_tile, then packs the single reduced tile to c_16. Regression baseline
// for reduce_block_2_0.cpp (bit-identical output). The two kernels differ ONLY in the reduce loop: this one
// loops legacy reduce_tile, the 2_0 one calls experimental::reduce_block once. The CB must be >= num_tiles
// deep (wait_front(num_tiles) keeps the whole block resident) -- the TEST_F passes cb_depth_tiles=num_tiles.
void kernel_main() {
    std::uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr auto cb_data = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    CircularBuffer cb0(cb_data);
    CircularBuffer cb1(cb_scaler);
    CircularBuffer cb16(cb_out);

    compute_kernel_hw_startup(cb_data, cb_scaler, cb_out);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_data, cb_scaler, cb_out);

    cb0.wait_front(num_tiles);
    cb1.wait_front(num_tiles);
    cb16.reserve_back(1);

    tile_regs_acquire();
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_data, cb_scaler, i, 0, 0);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb0.pop_front(num_tiles);
    cb1.pop_front(num_tiles);
    cb16.push_back(1);

    reduce_uninit();
}
