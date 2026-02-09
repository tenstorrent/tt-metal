// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/mul_reduce_scalar.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);    // Input A
    experimental::CircularBuffer cb1(tt::CBIndex::c_1);    // Input B
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);  // Output (reduced)

    // Initialize hardware before any operations
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    // Wait for num_tiles from each input
    cb0.wait_front(num_tiles);
    cb1.wait_front(num_tiles);

    // Reserve space for output
    cb16.reserve_back(1);

    // Initialize the multiply + reduce scalar operation
    ckernel::mul_reduce_scalar_init(tt::CBIndex::c_0, tt::CBIndex::c_1);

    tile_regs_acquire();

    // Perform fused multiply + reduce scalar
    ckernel::mul_reduce_scalar_tile<REDUCE_OP>(tt::CBIndex::c_0, tt::CBIndex::c_1, num_tiles);

    tile_regs_commit();
    tile_regs_wait();

    // Pack the result (MUST be tile index 0)
    pack_tile(0, tt::CBIndex::c_16);

    tile_regs_release();

    // Release input tiles (ready for future multiple tiles support)
    cb0.pop_front(num_tiles);
    cb1.pop_front(num_tiles);

    // Push output tile
    cb16.push_back(1);

    // Uninitialize the operation
    ckernel::mul_reduce_scalar_uninit();
}
