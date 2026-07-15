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
    const uint32_t dram_buffer_size = get_arg(args::dram_buffer_size);
    const uint32_t dram_src_bank_id = get_arg(args::dram_bank_id);
    const uint32_t signal_value = get_arg(args::signal_value);

    Noc noc;
    CoreLocalMem<std::uint32_t> l1_buffer(l1_dst_address);
    AllocatorBank<AllocatorBankType::DRAM> src_dram;
    Semaphore semaphore(sem::sem);

    semaphore.wait(signal_value);

    DPRINT(
        "Reading {} bytes from DRAM address {} in bank {} and writing it to L1 address {}\n",
        dram_buffer_size,
        dram_src_address,
        dram_src_bank_id,
        l1_dst_address);

    noc.async_read(src_dram, l1_buffer, dram_buffer_size, {.bank_id = dram_src_bank_id, .addr = dram_src_address}, {});
    noc.async_read_barrier();

    semaphore.up(1);
}
