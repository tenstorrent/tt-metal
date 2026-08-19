// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM producer kernel for BenchmarkCaseSixDebug.
//
// Single 1Sx1A DFB (logical id 0): DM4 (t0) → Neo0 (t0).
// Uses explicit reserve_back / push_back (implicit_sync=false) so producer
// finish() does not spin in handle_final_credits WTP1 waiting for ISR-posted
// producer TC credits. Matches the passing DMTensixTest1xDFB1Sx1S producer path.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    uint32_t dm_id;
    asm volatile("csrr %0, mhartid" : "=r"(dm_id));

    if (dm_id != 4) {
        return;
    }

    Noc noc;
    DataflowBuffer dfb(0);
    const uint32_t entry_size = dfb.get_entry_size();

    AllocatorBank<AllocatorBankType::DRAM> dram{};
    dfb.reserve_back(1);
    noc.async_read(dram, dfb, entry_size, {.bank_id = 0, .addr = 0}, {});
    noc.async_read_barrier();
    dfb.push_back(1);
    dfb.finish();
}
