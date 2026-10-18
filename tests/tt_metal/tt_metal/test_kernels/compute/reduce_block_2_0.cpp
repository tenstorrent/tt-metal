// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"  // compute_kernel_hw_startup, tile_regs_*, PoolType/ReduceDim
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/reduce.h"
#include "api/compute/experimental/2_0/pack.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"
#include "api/compute/experimental/2_0/hw_startup.h"

// Id-free (2.0) REDUCE_SCALAR SUM block kernel, classic circular buffers. One LLKOperand per input
// (data, scaler) + output; reduce is format-free so only geometry + addresses flow through. Reduces num_tiles
// data tiles via a SINGLE experimental::reduce_block call with N-in / N-out semantics (data tile i -> DST[i]),
// then packs the num_tiles reduced tiles to c_16. Output must be bit-identical to reduce_block_legacy.cpp
// (which calls the legacy reduce_block). Differs from that kernel ONLY in the reduce call, isolating
// reduce_block. The CBs must be >= num_tiles deep -- the TEST_F passes cb_depth_tiles=num_tiles.
void kernel_main() {
    std::uint32_t num_tiles = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb1(tt::CBIndex::c_1);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto data_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto scaler_cb = experimental::Cb<tt::CBIndex::c_1>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto data_desc = experimental::to_llk_mem_descriptor(data_cb);
    constexpr auto scaler_desc = experimental::to_llk_mem_descriptor(scaler_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using DataOp = experimental::LLKOperand<static_cast<DataFormat>(data_desc.format), data_desc.shape>;
    using ScalerOp = experimental::LLKOperand<static_cast<DataFormat>(scaler_desc.format), scaler_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    compute_kernel_hw_startup(
        DataOp(data_cb.read_address()), ScalerOp(scaler_cb.read_address()), OutOp(out_cb.write_address()));
    experimental::reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(
        DataOp(data_cb.read_address()), OutOp(out_cb.write_address()));

    cb0.wait_front(num_tiles);
    cb1.wait_front(num_tiles);
    cb16.reserve_back(num_tiles);

    tile_regs_acquire();
    experimental::reduce_block<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(
        DataOp(data_cb.read_address()), ScalerOp(scaler_cb.read_address()), 0, 0, 0, num_tiles);
    tile_regs_commit();

    tile_regs_wait();
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        experimental::pack_tile(OutOp(out_cb.write_address()), i, i);
    }
    tile_regs_release();

    cb0.pop_front(num_tiles);
    cb1.pop_front(num_tiles);
    cb16.push_back(num_tiles);

    experimental::reduce_uninit();
}
