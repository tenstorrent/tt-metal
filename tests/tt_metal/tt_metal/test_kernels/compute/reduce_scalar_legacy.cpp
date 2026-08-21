// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) minimal REDUCE_SCALAR SUM kernel, classic circular buffers: c_0 = data, c_1 = scaler,
// c_16 = reduced output. Regression baseline for reduce_scalar_idfree.cpp (bit-identical output). The
// shipping classic-CB reduce kernels (rmsnorm/layernorm/max_pool) are full fused ops, so — as with
// tilize/untilize — a minimal dedicated baseline is used for the differential test.
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
