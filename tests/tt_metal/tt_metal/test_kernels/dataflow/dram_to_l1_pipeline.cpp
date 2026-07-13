// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t dram_src_address = get_arg(args::dram_addr);
    const uint32_t l1_dst_address = get_arg(args::l1_addr);
    const uint32_t num_elements = get_arg(args::num_elements);
    const uint32_t dram_src_bank_id = get_arg(args::dram_bank_id);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> src_dram;
    Semaphore sem(sem::sem);
#ifdef WAIT_FOR_REMOTE_SEM
    Semaphore remote_sem(sem::remote_sem);
#endif

    for (uint32_t i = 0; i < num_elements; i++) {
#ifdef WAIT_FOR_REMOTE_SEM
        remote_sem.down(1);
#endif

        const uint32_t offset = i * static_cast<uint32_t>(sizeof(uint32_t));
        DPRINT(
            "Reading {} bytes from DRAM address {} in bank {} and writing it to L1 address {}\n",
            sizeof(uint32_t),
            dram_src_address + offset,
            dram_src_bank_id,
            l1_dst_address + offset);
        CoreLocalMem<uint32_t> l1_buf(l1_dst_address + offset);
        noc.async_read(
            src_dram, l1_buf, sizeof(uint32_t), {.bank_id = dram_src_bank_id, .addr = dram_src_address + offset}, {});
        noc.async_read_barrier();

        sem.up(1);
    }
}
