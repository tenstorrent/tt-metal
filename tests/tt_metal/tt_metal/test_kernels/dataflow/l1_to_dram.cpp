// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    const uint32_t dram_dst_address = get_arg_val<uint32_t>(0);
    const uint32_t l1_src_address = get_arg_val<uint32_t>(1);
    const uint32_t dram_buffer_size = get_arg_val<uint32_t>(2);
    const uint32_t dram_dst_bank_id = get_arg_val<uint32_t>(3);
    const uint32_t signal_value = get_arg_val<uint32_t>(4);

    Noc noc;
    CoreLocalMem<std::uint32_t> l1_buffer(l1_src_address);
    AllocatorBank<AllocatorBankType::DRAM> dst_dram;
    Semaphore semaphore(get_compile_time_arg_val(0));

    semaphore.wait(signal_value);

    DPRINT << "Reading " << dram_buffer_size << " bytes from L1 address " << l1_src_address
           << " and writing it to DRAM address " << dram_dst_address << " in bank " << dram_dst_bank_id << ENDL();
    DEVICE_PRINT(
        "Reading {} bytes from L1 address {} and writing it to DRAM address {} in bank {}\n",
        dram_buffer_size,
        l1_src_address,
        dram_dst_address,
        dram_dst_bank_id);

    noc.async_write(l1_buffer, dst_dram, dram_buffer_size, {}, {.bank_id = dram_dst_bank_id, .addr = dram_dst_address});
    noc.async_write_barrier();

    semaphore.up(1);
}
