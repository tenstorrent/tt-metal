// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/sum_reduce_scalar.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const float scaler = __builtin_bit_cast(float, get_arg_val<uint32_t>(1));

    CircularBuffer cb0(tt::CBIndex::c_0);    // Input
    CircularBuffer cb16(tt::CBIndex::c_16);  // Output (reduced)

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb0.wait_front(num_tiles);
    cb16.reserve_back(1);

    ckernel::sum_reduce_scalar_init(tt::CBIndex::c_0);

    tile_regs_acquire();

    ckernel::sum_reduce_scalar_tile(tt::CBIndex::c_0, tt::CBIndex::c_16, num_tiles, scaler);

    tile_regs_commit();
    tile_regs_wait();

    // The reduced scalar lands in tile index 0.
    pack_tile(0, tt::CBIndex::c_16);

    tile_regs_release();

    cb0.pop_front(num_tiles);
    cb16.push_back(1);

    // sum_reduce_scalar shares the reduce tail, and therefore the teardown, with mul_reduce_scalar.
    ckernel::mul_reduce_scalar_uninit();
}
