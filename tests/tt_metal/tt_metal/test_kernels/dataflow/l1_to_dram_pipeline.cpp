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
    const uint32_t dram_dst_address = get_arg(args::dram_addr);
    const uint32_t l1_src_address = get_arg(args::l1_addr);
    const uint32_t num_elements = get_arg(args::num_elements);
    const uint32_t dram_dst_bank_id = get_arg(args::dram_bank_id);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dst_dram;
    Semaphore sem(sem::sem);
#ifdef INCREMENT_REMOTE_SEM
    Semaphore remote_sem(sem::remote_sem);
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);
#endif

    for (uint32_t i = 0; i < num_elements; i++) {
        sem.down(1);

        const uint32_t offset = i * static_cast<uint32_t>(sizeof(uint32_t));
        DPRINT(
            "Reading {} bytes from L1 address {} and writing it to DRAM address {} in bank {}\n",
            sizeof(uint32_t),
            l1_src_address + offset,
            dram_dst_address + offset,
            dram_dst_bank_id);
        CoreLocalMem<uint32_t> l1_buf(l1_src_address + offset);
        noc.async_write(
            l1_buf, dst_dram, sizeof(uint32_t), {}, {.bank_id = dram_dst_bank_id, .addr = dram_dst_address + offset});
        noc.async_write_barrier();

#ifdef INCREMENT_REMOTE_SEM
        remote_sem.up(noc, remote_noc_x, remote_noc_y, 1);
#endif
    }
}
