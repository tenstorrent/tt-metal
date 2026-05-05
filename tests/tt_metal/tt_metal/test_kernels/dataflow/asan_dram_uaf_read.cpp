// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ASan negative test kernel: reads one cache-line-sized chunk from a DRAM
// bank-relative address that has already been freed host-side
// (MeshBuffer destroyed before EnqueueProgram). AllocatorImpl::deallocate_buffer
// reaches __emule_buffer_free, which repoisons the freed bank pages. Under
// ASan, the noc_async_read's host-side memcpy from the freed region traps.
//
// Runtime args:
//   0: captured DRAM bank-relative address of the freed buffer (uint32_t)
//   1: bank id to address (uint32_t)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t freed_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    constexpr uint32_t kReadBytes = 64;

    // Reserve a tiny in-bounds scratch region in L1 to receive the read.
    constexpr uint32_t kScratchAddr = 0x10000;
    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(bank_id, freed_addr);
    noc_async_read(noc_addr, kScratchAddr, kReadBytes);
    noc_async_read_barrier();
}
