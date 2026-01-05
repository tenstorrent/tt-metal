// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dram_dst_address = get_arg_val<uint32_t>(0);
    const uint32_t l1_src_address = get_arg_val<uint32_t>(1);
    const uint32_t dram_buffer_size = get_arg_val<uint32_t>(2);
    const uint32_t dram_dst_bank_id = get_arg_val<uint32_t>(3);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(4);
    const uint32_t semaphore_value = get_arg_val<uint32_t>(5);

    experimental::Noc noc;
    experimental::CoreLocalMem<std::uint32_t> l1_buffer(l1_src_address);
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dst_dram;

    DPRINT << "dram_dst_address: " << dram_dst_address << ENDL();

    experimental::Semaphore semaphore(semaphore_id);
    semaphore.wait(semaphore_value);

    noc.async_write(
        l1_buffer, dst_dram, dram_buffer_size, {}, {.bank_id = dram_dst_bank_id, .addr = dram_dst_address});
    noc.async_write_barrier();

    DPRINT << "l1_src_address: " << l1_src_address << ENDL();

    semaphore.up(1);
}
