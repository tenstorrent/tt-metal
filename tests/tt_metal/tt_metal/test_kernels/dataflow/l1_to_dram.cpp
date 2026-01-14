// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/noc.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    const uint32_t dram_dst_address = get_arg_val<uint32_t>(0);
    const uint32_t l1_src_address = get_arg_val<uint32_t>(1);
    const uint32_t dram_buffer_size = get_arg_val<uint32_t>(2);
    const uint32_t dram_dst_bank_id = get_arg_val<uint32_t>(3);
    const uint32_t signal_value = get_arg_val<uint32_t>(4);

    experimental::Noc noc;
    experimental::CoreLocalMem<std::uint32_t> l1_buffer(l1_src_address);
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dst_dram;

    // print out all arguments with names
    DPRINT << "dram_dst_address: " << dram_dst_address << ENDL();
    DPRINT << "l1_src_address: " << l1_src_address << ENDL();
    DPRINT << "dram_buffer_size: " << dram_buffer_size << ENDL();
    DPRINT << "dram_dst_bank_id: " << dram_dst_bank_id << ENDL();
    DPRINT << "signal_value: " << signal_value << ENDL();

    volatile tt_l1_ptr std::uint32_t* signal_addr = (tt_l1_ptr uint32_t*)(MEM_L1_UNCACHED_BASE);
    while (*signal_addr != signal_value) {
        DPRINT << "signal_addr: " << *signal_addr << ENDL();
    }

    DPRINT << "before write" << ENDL();

    noc.async_write(
        l1_buffer, dst_dram, dram_buffer_size, {}, {.bank_id = dram_dst_bank_id, .addr = dram_dst_address});
    DPRINT << "after write" << ENDL();
    noc.async_write_barrier<experimental::Noc::BarrierMode::TXN_ID>();
    DPRINT << "after write barrier" << ENDL();

    DPRINT << "signal_addr before update: " << *signal_addr << ENDL();
    *signal_addr = signal_value + 1;
    DPRINT << "signal_addr after update: " << *signal_addr << ENDL();

    DPRINT << 77777 << ENDL();
}
