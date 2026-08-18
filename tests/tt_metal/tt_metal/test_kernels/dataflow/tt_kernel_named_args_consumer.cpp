// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: TT_KERNEL named-args loopback consumer (data movement), "1st world args" syntax.
//
// The TT_KERNEL counterpart to named_args_loopback_consumer.cpp: reads a DFB and writes it to a
// single-page DRAM buffer. Arguments are the entry's parameters; genfiles generates kernel_main().
//
// Args (no varargs — the TT_KERNEL syntax doesn't express them):
//   bank_id     — CTA (template), DRAM bank ID
//   entry_size  — CTA (template), bytes per DFB entry
//   dst_addr    — RTA (function param, per-node), DRAM destination address
//   num_entries — CRTA (function param, broadcast), number of entries to transfer
//
// Paired with tt_kernel_named_args_producer.cpp; the host verifies input == output.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"  // provides TT_KERNEL, get_arg, the args:: / dfb:: accessors

template <uint32_t bank_id, uint32_t entry_size>  // CTAs (compile-time)
TT_KERNEL void loopback_consumer(
    uint32_t dst_addr,       // RTA (per-node)
    uint32_t num_entries) {  // CRTA (broadcast)
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram_dst;
    DataflowBuffer buf(dfb::loopback_dfb);

    for (uint32_t i = 0; i < num_entries; i++) {
        buf.wait_front(1);
        noc.async_write(buf, dram_dst, entry_size, {}, {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        buf.pop_front(1);
        dst_addr += entry_size;  // by-value function param; mutating the local copy is fine
    }
}
