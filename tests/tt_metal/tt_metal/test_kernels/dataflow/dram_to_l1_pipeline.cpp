// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/noc.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "experimental/noc_semaphore.h"

void kernel_main() {
    const uint32_t dram_src_address = get_arg_val<uint32_t>(0);
    const uint32_t l1_dst_address = get_arg_val<uint32_t>(1);
    const uint32_t num_elements = get_arg_val<uint32_t>(2);
    const uint32_t dram_src_bank_id = get_arg_val<uint32_t>(3);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> src_dram;
    experimental::Semaphore sem(get_compile_time_arg_val(0));
#ifdef WAIT_FOR_REMOTE_SEM
    experimental::Semaphore remote_sem(get_compile_time_arg_val(1));
#endif

    for (uint32_t i = 0; i < num_elements; i++) {
#ifdef WAIT_FOR_REMOTE_SEM
        remote_sem.down(1);
#endif

        const uint32_t offset = i * static_cast<uint32_t>(sizeof(uint32_t));
        DPRINT << "Reading " << sizeof(uint32_t) << " bytes from DRAM address " << dram_src_address + offset
               << " in bank " << dram_src_bank_id << " and writing it to L1 address " << l1_dst_address + offset
               << ENDL();
        experimental::CoreLocalMem<uint32_t> l1_buf(l1_dst_address + offset);
        noc.async_read(
            src_dram, l1_buf, sizeof(uint32_t), {.bank_id = dram_src_bank_id, .addr = dram_src_address + offset}, {});
        noc.async_read_barrier();

        sem.up(1);
    }
}
