// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/reduce.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/compute/experimental/2_0/tilize.h"
#include "api/dataflow/circular_buffer.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Two tiled Float16_b tiles, no reconfig and no uninit between ops.
// Tile 0: tilize_init (leaves tilize mode) then copy_init + pack_init must repair it.
// Tile 1: reduce_init (leaves the row edge mask) then copy_init + pack_init must repair it.
// Golden is the identity: both outputs equal their inputs.
void kernel_main() {
    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto out_cb = experimental::Cb<tt::CBIndex::c_16>{};
    constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);
    constexpr auto out_desc = experimental::to_llk_mem_descriptor(out_cb);
    using InOp = experimental::LLKOperand<static_cast<DataFormat>(in_desc.format), in_desc.shape>;
    using OutOp = experimental::LLKOperand<static_cast<DataFormat>(out_desc.format), out_desc.shape>;

    const InOp in(in_cb.read_address());
    const OutOp out(out_cb.write_address());

    compute_kernel_hw_startup(in, out);

    experimental::tilize_init(in, 1 /*block*/, out);
    experimental::copy_init(in);
    experimental::pack_init(out);

    cb0.wait_front(1);
    cb16.reserve_back(1);
    tile_regs_acquire();
    experimental::copy_tile(InOp(in_cb.read_address()), 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    experimental::pack_tile(OutOp(out_cb.write_address()), 0, 0);
    tile_regs_release();
    cb0.pop_front(1);
    cb16.push_back(1);

    experimental::reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(in, out);
    experimental::copy_init(in);
    experimental::pack_init(out);

    cb0.wait_front(1);
    cb16.reserve_back(1);
    tile_regs_acquire();
    experimental::copy_tile(InOp(in_cb.read_address()), 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    experimental::pack_tile(OutOp(out_cb.write_address()), 0, 0);
    tile_regs_release();
    cb0.pop_front(1);
    cb16.push_back(1);
}
