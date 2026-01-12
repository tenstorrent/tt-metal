// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/noc_semaphore.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    const uint32_t dram_src_address = get_arg_val<uint32_t>(0);
    const uint32_t l1_dst_address = get_arg_val<uint32_t>(1);
    const uint32_t dram_buffer_size = get_arg_val<uint32_t>(2);
    const uint32_t dram_src_bank_id = get_arg_val<uint32_t>(3);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(4);
    const uint32_t semaphore_value = get_arg_val<uint32_t>(5);

    experimental::Noc noc;
    experimental::CoreLocalMem<std::uint32_t> l1_buffer(l1_dst_address);
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> src_dram;

    // print out all arguments with names
    //  DPRINT << "dram_src_address: " << dram_src_address << ENDL();
    //  DPRINT << "l1_dst_address: " << l1_dst_address << ENDL();
    //  DPRINT << "dram_buffer_size: " << dram_buffer_size << ENDL();
    //  DPRINT << "dram_src_bank_id: " << dram_src_bank_id << ENDL();
    //  DPRINT << "semaphore_id: " << semaphore_id << ENDL();
    //  DPRINT << "semaphore_value: " << semaphore_value << ENDL();

    experimental::Semaphore semaphore(semaphore_id);
    semaphore.wait(semaphore_value);

    noc.async_read(
        src_dram, l1_buffer, dram_buffer_size, {.bank_id = dram_src_bank_id, .addr = dram_src_address}, {});
    noc.async_read_barrier();

    semaphore.set(1);

    DPRINT << 9999 << ENDL();
}
