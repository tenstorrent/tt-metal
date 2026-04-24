// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: Metal 2.0 named-args loopback producer.
// Reads data from a single-page DRAM buffer and pushes it entry-by-entry into a DFB.
//
// Exercises the Metal 2.0 kernel-args feature surface:
//   args::src_addr     — named RTA (per-node), DRAM source address
//   args::num_entries  — named CRTA (broadcast), number of entries to transfer
//   args::bank_id      — named CTA, DRAM bank ID (typically 0 for single-page buffers)
//   args::entry_size   — named CTA, bytes per DFB entry
//   get_vararg(0..2)   — three RTA varargs (per-node, positional from 0)
//   get_common_vararg(0) — one CRTA vararg (broadcast, positional from 0)
//
// Vararg offset verification: the XOR of this kernel's four vararg values is folded into the
// first word of each DFB entry before push_back. The consumer does the symmetric thing,
// XORing its own vararg sum into the first word on read. The host arranges for both sums to
// match, so the two XORs cancel and the input/output data compare stays clean — but only if
// every get_vararg / get_common_vararg offset is correct. A wrong offset at any index will
// return the wrong word, flip bits in the sum, scramble the first word, and fail the test.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Exercise all three type-deduction forms for get_arg on HW:
    //   - plain `auto`: RTA, deduced to uint32_t (today)
    //   - `const auto`: CRTA, non-constexpr runtime value
    //   - `constexpr auto`: CTAs only — the RTA/CRTA overloads aren't constexpr because
    //     their values aren't known until dispatch time
    auto src_addr = get_arg(args::src_addr);
    const auto num_entries = get_arg(args::num_entries);
    constexpr auto bank_id = get_arg(args::bank_id);
    auto entry_size = get_arg(args::entry_size);

    // Vararg sum: exercises get_vararg(0), get_vararg(1), get_vararg(2), get_common_vararg(0).
    // If any offset is wrong, this XOR won't cancel against the consumer's and the first
    // word of each DFB entry will end up corrupted in the DRAM output buffer.
    const uint32_t vararg_xor = get_vararg(0) ^ get_vararg(1) ^ get_vararg(2) ^ get_common_vararg(0);

    experimental::DataflowBuffer buf(dfb::loopback_dfb);

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.reserve_back(1);
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);
        noc_async_read(src_noc_addr, buf.get_write_ptr(), entry_size);
        noc_async_read_barrier();
        // Fold vararg sum into the first word of this entry — the consumer will XOR
        // its own sum into the same word on read.
        volatile tt_l1_ptr uint32_t* dfb_write_ptr = (volatile tt_l1_ptr uint32_t*)buf.get_write_ptr();
        dfb_write_ptr[0] ^= vararg_xor;
        buf.push_back(1);
        src_addr += entry_size;
    }
}
