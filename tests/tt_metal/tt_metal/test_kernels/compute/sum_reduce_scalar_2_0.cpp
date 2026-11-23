// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/sum_reduce_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// sum_reduce_scalar.cpp over the 2.0 host API (DataflowBuffer bindings, named compile-time args).
void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);
    const float scaler = __builtin_bit_cast(float, static_cast<uint32_t>(get_arg(args::scaler_bits)));

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t in = dfb_in.get_id();
    const uint32_t out = dfb_out.get_id();

    compute_kernel_hw_startup(in, out);

    dfb_in.wait_front(num_tiles);
    dfb_out.reserve_back(1);

    ckernel::sum_reduce_scalar_init(in);

    tile_regs_acquire();
    ckernel::sum_reduce_scalar_tile(in, out, num_tiles, scaler);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, out);  // The reduced scalar lands in tile 0.
    tile_regs_release();

    dfb_in.pop_front(num_tiles);
    dfb_out.push_back(1);

    // sum_reduce_scalar shares the reduce tail, and therefore the teardown, with mul_reduce_scalar.
    ckernel::mul_reduce_scalar_uninit();
}
