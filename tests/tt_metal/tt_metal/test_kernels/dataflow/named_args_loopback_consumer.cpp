// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 named-args loopback consumer.
// Reads data from a DFB and writes it to a single-page DRAM buffer.
//
// Exercises the Metal 2.0 kernel-args feature surface:
//   args::dst_addr              — named RTA (per-node), DRAM destination address
//   args::num_entries           — named CRTA (broadcast), number of entries to transfer
//   args::bank_id               — named CTA, DRAM bank ID
//   args::entry_size            — named CTA, bytes per DFB entry
//   get_vararg(0..1)            — two RTA varargs (different count from producer — verifies
//                                 that the baked-in offset is per-kernel, not shared state)
//   get_common_vararg(0)        — one CRTA vararg
//
// Uses the default `args` namespace; the producer uses a custom `producer_args` namespace.
// See the producer source for the XOR-cancellation verification trick.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t dst_addr = get_arg(args::dst_addr);
    uint32_t num_entries = get_arg(args::num_entries);
    constexpr uint32_t bank_id = get_arg(args::bank_id);
    constexpr uint32_t entry_size = get_arg(args::entry_size);

    // Vararg sum: exercises get_vararg(0), get_vararg(1), get_common_vararg(0). Host arranges
    // for this to equal the producer's sum; the two XORs cancel and the first word of each
    // entry survives round-trip. A wrong offset in either kernel breaks the cancellation.
    const uint32_t vararg_xor = get_vararg(0) ^ get_vararg(1) ^ get_common_vararg(0);

    experimental::DataflowBuffer buf(dfb::loopback_dfb);

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.wait_front(1);
        // Unfold the producer's XOR by applying our (equal) sum to the same word.
        volatile tt_l1_ptr uint32_t* dfb_read_ptr = (volatile tt_l1_ptr uint32_t*)buf.get_read_ptr();
        dfb_read_ptr[0] ^= vararg_xor;
        uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);
        noc_async_write(buf.get_read_ptr(), dst_noc_addr, entry_size);
        noc_async_write_barrier();
        buf.pop_front(1);
        dst_addr += entry_size;
    }
}
