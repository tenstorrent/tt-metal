// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: TT_KERNEL named-args loopback producer (data movement), "1st world args" syntax.
//
// The TT_KERNEL counterpart to named_args_loopback_producer.cpp: the arguments are the entry's
// parameters (CTAs as template parameters, RTA/CRTA as function parameters) and genfiles
// generates the kernel_main() shim. Reads data from a single-page DRAM buffer and pushes it
// entry-by-entry into a DFB.
//
// Args (no varargs — the TT_KERNEL syntax doesn't express them):
//   bank_id     — CTA (template), DRAM bank ID
//   entry_size  — CTA (template), bytes per DFB entry
//   src_addr    — RTA (function param, per-node), DRAM source address
//   num_entries — CRTA (function param, broadcast), number of entries to transfer
//
// Verification is by data round-trip: the companion consumer copies the DFB back out to DRAM,
// and the host compares input == output. A wrong binding for any of the four args (wrong
// address, size, bank, or count) corrupts the round-trip and fails the test.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"  // provides TT_KERNEL, get_arg, the args:: / dfb:: accessors

template <uint32_t bank_id, uint32_t entry_size>  // CTAs (compile-time)
TT_KERNEL void loopback_producer(
    uint32_t src_addr,       // RTA (per-node)
    uint32_t num_entries) {  // CRTA (broadcast)
    DataflowBuffer buf(dfb::loopback_dfb);

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);
        noc_async_read(src_noc_addr, buf.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        buf.push_back(1);
        src_addr += entry_size;  // by-value function param; mutating the local copy is fine
    }
}
