// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Chunked counterpart of mul_reduce_scalar.cpp.
//
// mul_reduce_scalar_tile requires every product to be resident in DST before
// the reduce consumes it, so it caps num_tiles at the DST capacity.
// mul_reduce_scalar_chunked_tile lifts that cap: it reserves
// DST[dst_capacity - 1] as a cross-chunk accumulator, stages products in the
// remaining slots, and reduces one chunk at a time. Chunking only reorders the
// accumulation, so the golden is the same sum(A*B) the non-chunked test uses.
//
// num_tiles and dst_capacity are template parameters of the compute API, so
// they arrive as compile-time defines rather than runtime args.

#include <cstdint>

#include "api/compute/experimental/rmsnorm.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = CHUNKED_NUM_TILES;
    constexpr uint32_t dst_capacity = CHUNKED_DST_CAPACITY;
    // Documented by the compute API: on return the scaled sum of products is in
    // DST[dst_capacity - 1], not DST[0].
    constexpr uint32_t accumulator = dst_capacity - 1;

    CircularBuffer cb0(tt::CBIndex::c_0);    // Input A
    CircularBuffer cb1(tt::CBIndex::c_1);    // Input B
    CircularBuffer cb16(tt::CBIndex::c_16);  // Output (reduced)

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    cb0.wait_front(num_tiles);
    cb1.wait_front(num_tiles);
    cb16.reserve_back(1);

    // The compute API's contract: the caller initializes mul_reduce_scalar AND
    // add_binary (the cross-chunk accumulate uses add_binary_tile), then
    // acquires DST.
    ckernel::mul_reduce_scalar_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    ckernel::add_binary_tile_init();

    tile_regs_acquire();

    ckernel::mul_reduce_scalar_chunked_tile<num_tiles, dst_capacity>(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    tile_regs_commit();
    tile_regs_wait();

    // The reduce pack mask is still configured, which is what makes element [0]
    // the only defined lane; uninit comes after the pack.
    pack_tile(accumulator, tt::CBIndex::c_16);

    tile_regs_release();

    cb0.pop_front(num_tiles);
    cb1.pop_front(num_tiles);
    cb16.push_back(1);

    ckernel::mul_reduce_scalar_uninit();
}
