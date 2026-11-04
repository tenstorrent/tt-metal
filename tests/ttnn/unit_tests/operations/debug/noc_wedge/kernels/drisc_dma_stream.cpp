// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Traffic-only DRISC (DRAM-core) kernel for the Blackhole NoC wedge reproducers: streams GDDR -> DRISC L1 DMA reads on
// one TX stream with `depth` reads in flight, optionally forwarding every landed block to Tensix L1 over the NoC (the
// flat routed expert's DRISC weight senders). Fixed counts, no semaphores; the only waits are this core's own DMA
// queue and NoC write flushes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/gddr_dma.h"

void kernel_main() {
    uint32_t a = 0;
    const uint32_t iters = get_arg_val<uint32_t>(a++);
    const uint32_t src_gddr = get_arg_val<uint32_t>(a++);  // bank-local GDDR address of this bank's scratch
    const uint32_t region = get_arg_val<uint32_t>(a++);    // bytes of GDDR scratch to cycle through
    const uint32_t bytes = get_arg_val<uint32_t>(a++);     // bytes per DMA read (multiple of 16)
    const uint32_t l1 = get_arg_val<uint32_t>(a++);        // DRISC L1 ring: depth x bytes
    const uint32_t stream = get_arg_val<uint32_t>(a++);    // TX stream 0 or 1
    const uint32_t depth = get_arg_val<uint32_t>(a++);     // DMA reads in flight
    const uint32_t forward = get_arg_val<uint32_t>(a++);   // 1: NoC-write every landed block to a Tensix core
    const uint32_t land = get_arg_val<uint32_t>(a++);      // Tensix L1 landing (4 slots of bytes)
    const uint32_t n_targets = get_arg_val<uint32_t>(a++);
    const uint32_t t0 = a;  // packed virtual x << 16 | y of each target

    const uint32_t slots = region / bytes;
    for (uint32_t i = 0; i < iters; ++i) {
        if (i >= depth) {
            experimental::dma_async_read_wait_n(stream, depth - 1);  // the oldest of `depth` landed (FIFO)
            if (forward) {
                const uint32_t xy = get_arg_val<uint32_t>(t0 + i % n_targets);
                noc_async_write(
                    l1 + (i % depth) * bytes, get_noc_addr(xy >> 16, xy & 0xFFFF, land + (i % 4) * bytes), bytes);
                noc_async_writes_flushed();  // (the slot is the next DMA's destination)
            }
        }
        experimental::dma_async_read(stream, src_gddr + (i % slots) * bytes, l1 + (i % depth) * bytes, bytes);
    }
    experimental::dma_async_read_barrier(stream);
    noc_async_write_barrier();
}
