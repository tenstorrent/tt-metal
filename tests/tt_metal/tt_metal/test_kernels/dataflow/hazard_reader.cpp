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
    // THROWAWAY (query-side POC): hand-emit the .tt.BUF_RW records for this kernel -- READ tensor::src
    // (slot 0) and WRITE tensor::dst (slot 1) -- so the host can confirm the reader's R/W set. SHT_NOTE,
    // non-alloc (empty flags): pure data, not loaded to device. Replaced later by inline-asm annotations
    // on the NoC read/write APIs.
    __asm__ volatile(
        ".pushsection .tt.BUF_RW,\"\",@note\n\t"
        ".4byte 0\n\t"  // slot 0 == tensor::src
        ".4byte 1\n\t"  // kind READ
        ".4byte 1\n\t"  // slot 1 == tensor::dst
        ".4byte 2\n\t"  // kind WRITE
        ".popsection");

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
