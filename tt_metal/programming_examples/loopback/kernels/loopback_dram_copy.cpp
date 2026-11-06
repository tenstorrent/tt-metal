// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/kernel_args.h"

void kernel_main() {
    // Read parameters from the kernel arguments. Arguments are read by the names declared in the host code.
    std::uint32_t l1_buffer_addr = get_arg(args::l1_buffer_addr);

    // Number of tiles to copy
    std::uint32_t num_tiles = get_arg(args::num_tiles);

    // Each tile is 32x32 elements of bfloat16, which is 2 bytes per element.
    // So the tile size in bytes is 32 * 32 * 2 = 2048 bytes.
    // Note that this is the same as the tile size used in the host code
    // when creating the buffers.
    // The source and destination buffers are tensors bound to this kernel by name in the host code.
    // The address and the layout of each tensor come from that binding.
    const auto in0 = TensorAccessor(tensor::src);
    const auto out0 = TensorAccessor(tensor::dst);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Issue a read to the NoC and write to the L1 buffer. This operation is asynchronous.
        // thus a barrier is needed to ensure that the read is complete before the write.
        noc_async_read_page(i, in0, l1_buffer_addr);
        noc_async_read_barrier();
        // Write back the tile to the destination DRAM buffer.
        // Again, this is an asynchronous operation, so we need a barrier to ensure the write
        // is complete before the next iteration.
        noc_async_write_page(i, out0, l1_buffer_addr);
        noc_async_write_barrier();
    }
}
