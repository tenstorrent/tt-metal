// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix Kernel: DRAM GDDR reads over NOC into Tensix L1

#include "experimental/core_local_mem.h"
#include "experimental/noc.h"
#include "experimental/endpoints.h"

void kernel_main() {
    const uint32_t src_dram_bank_id = get_arg_val<uint32_t>(0);
    const uint32_t src_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(2);
    const uint32_t total_bytes = get_arg_val<uint32_t>(3);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> bank;
    experimental::CoreLocalMem<uint32_t> dst(dst_l1_addr);

    noc.async_read(bank, dst, total_bytes, {.bank_id = src_dram_bank_id, .addr = src_dram_addr}, {});
    noc.async_read_barrier();
}
