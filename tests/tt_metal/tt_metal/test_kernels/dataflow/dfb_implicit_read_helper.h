// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared helper for DFB ISR-to-credit latency benchmarks.
//
// dfb_issue_implicit_read() issues one implicit-sync NOC read for a DFB
// producer, which will cause the ISR to fire once hw_threshold total reads
// have accumulated across all sharing DMs. Call this exactly per_txn times
// per DFB (per txn ID) to hit the threshold.
//
// Implementation details:
//   - The source is DRAM bank 0 at offset 0; data content is irrelevant since
//     the benchmark cares only about ISR timing, not data correctness.
//   - AllocatorBank<DRAM> resolves bank coordinates via the device-side
//     dram_bank_to_noc_xy[] table; no host-side address passing is needed.
//   - The DFB manages its txn ID, L1 destination (write_ptr), and entry size
//     internally; the caller provides only the Noc instance and DFB handle.

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

// Issues one implicit-sync NOC read for a DFB producer.
// Call per_txn times per DFB to trigger the ISR threshold.
FORCE_INLINE void dfb_issue_implicit_read(const Noc& noc, DataflowBuffer& dfb) {
    AllocatorBank<AllocatorBankType::DRAM> dram{};
    noc.async_read<NocOptions::TXN_ID>(dram, dfb, {.bank_id = 0, .addr = 0}, {});
}
