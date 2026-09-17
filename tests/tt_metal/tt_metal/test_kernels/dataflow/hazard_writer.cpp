// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Hazard-test WRITER (Metal 2.0 Host API / ProgramSpec path). Stages a known `pattern` into a
// framework-allocated node-local scratchpad, optionally busy-waits a deliberate stall (to widen the
// op-to-op RAW race window), then NoC-writes the scratchpad to the destination tensor via a
// TensorAccessor. The destination buffer is plumbed BY NAME through the tensor binding (tensor::dst),
// so the framework/detector can see this kernel WRITES that tensor. No addresses, no CB/DFB.
//
// The `Scratchpad`, `scratch::`, `tensor::`, and `args::` tokens are emitted by genfiles from the
// kernel's scratchpad/tensor/runtime-arg bindings; no manual include for those.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "c_tensix_core.h"

void kernel_main() {
    const uint32_t pattern = get_arg(args::pattern);
    const uint32_t stall_cycles = get_arg(args::stall);

    Scratchpad<uint32_t> pad(scratch::pad);
    TensorAccessor dst(tensor::dst);

    // Stage the known `pattern` into the framework-allocated scratchpad.
    const uint32_t n = pad.size();
    for (uint32_t i = 0; i < n; i++) {
        pad[i] = pattern;
    }
    asm("fence");

    // Deliberate stall BEFORE the write so the target lands late (widens the race window for a relaxed reader).
    if (stall_cycles) {
        const uint64_t end = c_tensix_core::read_wall_clock() + stall_cycles;
        while (c_tensix_core::read_wall_clock() < end) {
        }
    }

    Noc noc;
    noc.async_write(pad, dst, pad.size_in_bytes(), {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
}
