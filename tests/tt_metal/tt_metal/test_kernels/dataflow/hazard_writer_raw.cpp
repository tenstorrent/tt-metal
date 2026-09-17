// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Hazard-test WRITER on the intentionally UN-ANALYZABLE path (the "bail" case). Unlike hazard_writer,
// this kernel has NO tensor binding: its destination is a plain named runtime-arg carrying a raw DRAM
// address, written via the VANILLA C NoC free-functions (InterleavedAddrGen + noc_async_write) rather
// than a TensorAccessor. With no tensor binding the framework cannot infer which buffer this kernel
// touches, so a future op-to-op detector must conservatively KEEP the barrier. Its staging scratchpad
// is still a framework-allocated ScratchpadSpec binding -- no magic address anywhere.
//
// The `Scratchpad`, `scratch::`, and `args::` tokens are emitted by genfiles from the bindings.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "c_tensix_core.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg(args::dst_addr);  // plain address, NOT a tensor binding
    const uint32_t pattern = get_arg(args::pattern);
    const uint32_t stall_cycles = get_arg(args::stall);

    Scratchpad<uint32_t> pad(scratch::pad);

    const uint32_t n = pad.size();
    for (uint32_t i = 0; i < n; i++) {
        pad[i] = pattern;
    }
    asm("fence");  // ensure the L1 writes land before the NoC reads them

    if (stall_cycles) {
        const uint64_t end = c_tensix_core::read_wall_clock() + stall_cycles;
        while (c_tensix_core::read_wall_clock() < end) {
        }
    }

    // Vanilla C NoC path: build the interleaved-DRAM NoC address for page 0 and push the scratchpad out
    // with the free-function noc_async_write -- no Noc / AllocatorBank / TensorAccessor abstraction.
    InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = pad.size_in_bytes()};
    const uint64_t dst_noc_addr = dst.get_noc_addr(0);
    noc_async_write(pad.get_base_address(), dst_noc_addr, pad.size_in_bytes());
    noc_async_write_barrier();
}
