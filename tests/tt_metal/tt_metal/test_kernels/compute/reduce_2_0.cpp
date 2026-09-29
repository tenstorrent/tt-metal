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

// Id-free (2.0) minimal single-DST reduce kernel, classic circular buffers. One LLKOperand per input
// (data, scaler) + output; reduce is format-free so only geometry + addresses flow through. The pool and
// reduce dim are selected at compile time via get_compile_time_arg_val(1)=PoolType, (2)=ReduceDim, so this
// ONE kernel covers reduce_{scalar,row,col} SUM and reduce scalar MAX -- those differed only in those two
// template args. Output must be bit-identical to the matching reduce_*_legacy.cpp. (reduce_block, which has
// N-in/N-out block semantics + a different pack loop, is a separate kernel.)
namespace {
constexpr auto kPool = static_cast<PoolType>(get_compile_time_arg_val(1));
constexpr auto kDim = static_cast<ReduceDim>(get_compile_time_arg_val(2));
}  // namespace

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
    experimental::reduce_init<kPool, kDim>(DataOp(data_cb.read_address()), OutOp(out_cb.write_address()));

    cb0.wait_front(num_tiles);
    cb1.wait_front(num_tiles);
    cb16.reserve_back(1);

    tile_regs_acquire();
    for (std::uint32_t i = 0; i < num_tiles; ++i) {
        experimental::reduce_tile<kPool, kDim>(
            DataOp(data_cb.read_address()), ScalerOp(scaler_cb.read_address()), i, 0, 0);
    }
    tile_regs_commit();

    tile_regs_wait();
    experimental::pack_tile(OutOp(out_cb.write_address()), 0, 0);
    tile_regs_release();

    cb0.pop_front(num_tiles);
    cb1.pop_front(num_tiles);
    cb16.push_back(1);

    experimental::reduce_uninit();
}
