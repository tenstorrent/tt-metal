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
    const uint32_t dram_dst_address = get_arg_val<uint32_t>(0);
    const uint32_t l1_src_address = get_arg_val<uint32_t>(1);
    const uint32_t num_elements = get_arg_val<uint32_t>(2);
    const uint32_t dram_dst_bank_id = get_arg_val<uint32_t>(3);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dst_dram;
    experimental::Semaphore sem(get_compile_time_arg_val(0));
#ifdef INCREMENT_REMOTE_SEM
    experimental::Semaphore remote_sem(get_compile_time_arg_val(1));
    const uint32_t remote_noc_x = get_arg_val<uint32_t>(4);
    const uint32_t remote_noc_y = get_arg_val<uint32_t>(5);
#endif

    for (uint32_t i = 0; i < num_elements; i++) {
        sem.down(1);

        const uint32_t offset = i * static_cast<uint32_t>(sizeof(uint32_t));
        DPRINT << "Reading " << sizeof(uint32_t) << " bytes from L1 address " << l1_src_address + offset
               << " and writing it to DRAM address " << dram_dst_address + offset << " in bank " << dram_dst_bank_id
               << ENDL();
        DEVICE_PRINT(
            "Reading {} bytes from L1 address {} and writing it to DRAM address {} in bank {}\n",
            sizeof(uint32_t),
            l1_src_address + offset,
            dram_dst_address + offset,
            dram_dst_bank_id);
        experimental::CoreLocalMem<uint32_t> l1_buf(l1_src_address + offset);
        noc.async_write(
            l1_buf, dst_dram, sizeof(uint32_t), {}, {.bank_id = dram_dst_bank_id, .addr = dram_dst_address + offset});
        noc.async_write_barrier();

#ifdef INCREMENT_REMOTE_SEM
        remote_sem.up(noc, remote_noc_x, remote_noc_y, 1);
#endif
    }
}
