// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Hazard-test READER (Metal 2.0 Host API / ProgramSpec path). Optionally busy-waits a deliberate stall
// (to widen the op-to-op WAR race window), then NoC-reads the source tensor into a framework-allocated
// node-local scratchpad and NoC-writes it back out to the destination tensor, both via TensorAccessors.
// Both buffers are plumbed BY NAME through tensor bindings (tensor::src reads, tensor::dst writes), so
// the framework/detector can see this kernel READS src and WRITES dst. No addresses, no CB/DFB.
//
// The `Scratchpad`, `scratch::`, `tensor::`, and `args::` tokens are emitted by genfiles from the
// kernel's scratchpad/tensor/runtime-arg bindings; no manual include for those.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "c_tensix_core.h"

void kernel_main() {
    const uint32_t stall_cycles = get_arg(args::stall);

    Scratchpad<uint32_t> pad(scratch::pad);
    TensorAccessor src(tensor::src);
    TensorAccessor dst(tensor::dst);

    // Deliberate stall BEFORE the read so the read happens late (widens the window for a relaxed racing writer).
    if (stall_cycles) {
        const uint64_t end = c_tensix_core::read_wall_clock() + stall_cycles;
        while (c_tensix_core::read_wall_clock() < end) {
        }
    }

    Noc noc;
    noc.async_read(src, pad, pad.size_in_bytes(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    noc.async_write(pad, dst, pad.size_in_bytes(), {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
}
