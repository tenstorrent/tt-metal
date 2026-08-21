// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"         // compute_kernel_hw_startup, tile_regs_*, pack_tile
#include "api/compute/reduce_custom.h"  // legacy CB-id reduce_block_max_row (SDPA MAX-row)
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) reduce_custom == SDPA block MAX-row kernel, classic circular buffers: c_0 = data, c_1 = scaler,
// c_16 = reduced output. Reduces a block of num_tiles data tiles in the width dimension, taking the per-row MAX
// across the whole block into DST[0], then packs the single reduced tile to c_16. Regression baseline for
// reduce_custom_2_0.cpp (bit-identical output). The two kernels differ ONLY in the reduce_block_max_row
// init/op/uninit (CB-id vs id-free LLKOperand); hw_startup + pack_tile stay legacy CB-id in BOTH. The CB must be
// >= num_tiles deep (the whole block is unpacked at once) -- the TEST_F passes cb_depth_tiles=num_tiles.
void kernel_main() {
    constexpr std::uint32_t block_ct_dim = get_compile_time_arg_val(0);  // == num_tiles

    constexpr auto cb_data = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    CircularBuffer cb0(cb_data);
    CircularBuffer cb1(cb_scaler);
    CircularBuffer cb16(cb_out);

    compute_kernel_hw_startup(cb_data, cb_scaler, cb_out);
    ckernel::reduce_block_max_row_init<block_ct_dim>(cb_out);

    cb0.wait_front(block_ct_dim);
    cb1.wait_front(block_ct_dim);
    cb16.reserve_back(1);

    tile_regs_acquire();
    ckernel::reduce_block_max_row<block_ct_dim>(cb_data, cb_scaler, /*row_start_index=*/0, /*idst=*/0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb0.pop_front(block_ct_dim);
    cb1.pop_front(block_ct_dim);
    cb16.push_back(1);

    ckernel::reduce_block_max_row_uninit(cb_data);
}
