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
    const uint32_t dram_src_address = get_vararg(0);
    const uint32_t l1_dst_address = get_vararg(1);
    const uint32_t dram_buffer_size = get_vararg(2);
    const uint32_t dram_src_bank_id = get_vararg(3);
    const uint32_t signal_value = get_vararg(4);

    Noc noc;
    CoreLocalMem<std::uint32_t> l1_buffer(l1_dst_address);
    AllocatorBank<AllocatorBankType::DRAM> src_dram;
    Semaphore semaphore(sem::sem);

    semaphore.wait(signal_value);

    DPRINT << "Reading " << dram_buffer_size << " bytes from DRAM address " << dram_src_address << " in bank "
           << dram_src_bank_id << " and writing it to L1 address " << l1_dst_address << ENDL();
    DEVICE_PRINT(
        "Reading {} bytes from DRAM address {} in bank {} and writing it to L1 address {}\n",
        dram_buffer_size,
        dram_src_address,
        dram_src_bank_id,
        l1_dst_address);

    noc.async_read(src_dram, l1_buffer, dram_buffer_size, {.bank_id = dram_src_bank_id, .addr = dram_src_address}, {});
    noc.async_read_barrier();

    semaphore.up(1);
}
