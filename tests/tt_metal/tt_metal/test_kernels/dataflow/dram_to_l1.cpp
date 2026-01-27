// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/noc.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    const uint32_t dram_src_address = get_arg_val<uint32_t>(0);
    const uint32_t l1_dst_address = get_arg_val<uint32_t>(1);
    const uint32_t signal_address = get_arg_val<uint32_t>(2);
    const uint32_t dram_buffer_size = get_arg_val<uint32_t>(3);
    const uint32_t dram_src_bank_id = get_arg_val<uint32_t>(4);
    const uint32_t signal_value = get_arg_val<uint32_t>(5);

    experimental::Noc noc;
    experimental::CoreLocalMem<std::uint32_t> l1_buffer(l1_dst_address);
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> src_dram;

    volatile tt_l1_ptr std::uint32_t* signal_addr = (tt_l1_ptr uint32_t*)((uintptr_t)signal_address);
    while (*signal_addr != signal_value);

    DPRINT << "Reading " << dram_buffer_size << " bytes from DRAM address " << dram_src_address << " in bank "
           << dram_src_bank_id << " and writing it to L1 address " << l1_dst_address << ENDL();

    noc.async_read(src_dram, l1_buffer, dram_buffer_size, {.bank_id = dram_src_bank_id, .addr = dram_src_address}, {});
    noc.async_read_barrier();

    *signal_addr = signal_value + 1;
}
