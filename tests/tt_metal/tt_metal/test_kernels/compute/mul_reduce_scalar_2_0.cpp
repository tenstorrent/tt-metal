// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// mul_reduce_scalar.cpp over the 2.0 host API (DataflowBuffer bindings, named compile-time args).
void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t in0 = dfb_in0.get_id();
    const uint32_t in1 = dfb_in1.get_id();
    const uint32_t out = dfb_out.get_id();

    compute_kernel_hw_startup(in0, in1, out);

    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(num_tiles);
    dfb_out.reserve_back(1);

    ckernel::mul_reduce_scalar_init(in0, in1);

    tile_regs_acquire();
    ckernel::mul_reduce_scalar_tile<PoolType::SUM, DST_ACCUM_MODE>(in0, in1, out, num_tiles);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, out);  // The reduced scalar lands in tile 0.
    tile_regs_release();

    dfb_in0.pop_front(num_tiles);
    dfb_in1.pop_front(num_tiles);
    dfb_out.push_back(1);

    ckernel::mul_reduce_scalar_uninit();
}
