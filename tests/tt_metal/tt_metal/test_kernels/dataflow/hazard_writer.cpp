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
    // THROWAWAY (query-side POC): hand-emit one .tt.BUF_RW record -- (binding slot 0 == tensor::dst,
    // kind WRITE=2) -- to prove the host-side ELF parse + kernel lookup (Kernel::query_buf_rw) end to
    // end. SHT_NOTE, non-alloc: pure data, zero instructions, not loaded to device. This will be replaced
    // by inline-asm annotations on the NoC read/write APIs so the record is emitted from the actual
    // access, not by hand.
    __asm__ volatile(
        ".pushsection .tt.BUF_RW,\"\",@note\n\t"  // empty flags => non-alloc; SHT_NOTE. This GAS needs the
        ".4byte 0\n\t"                            // flags string before @type (bare ",@note" won't parse).
        ".4byte 2\n\t"                            // slot 0 == tensor::dst, kind WRITE
        ".popsection");

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
