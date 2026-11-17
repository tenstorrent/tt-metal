// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// For test_srca_forced_exp.cpp: one page each into c_0, c_1 and c_2 from consecutive bytes of one DRAM buffer.
// Runtime args: buffer address, then the three pages' bytes.

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    uint32_t addr = get_arg_val<uint32_t>(0);
    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    for (uint32_t i = 0; i < 3; i++) {
        const uint32_t bytes = get_arg_val<uint32_t>(1 + i);
        CircularBuffer cb(tt::CBIndex::c_0 + i);
        cb.reserve_back(1);
        noc.async_read(dram, cb, bytes, {.bank_id = 0, .addr = addr}, {});
        noc.async_read_barrier();
        cb.push_back(1);
        addr += bytes;
    }
}
