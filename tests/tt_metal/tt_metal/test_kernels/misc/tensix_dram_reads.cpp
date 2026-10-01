// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix Kernel: DRAM GDDR reads over NOC into Tensix L1

#include "api/core_local_mem.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    const uint32_t src_dram_bank_id = get_arg_val<uint32_t>(0);
    uint32_t src_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(2);
    const uint32_t total_bytes = get_arg_val<uint32_t>(3);
    const uint32_t iters = get_arg_val<uint32_t>(4);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> bank;
    CoreLocalMem<uint32_t> dst(dst_l1_addr);

    for (uint32_t i = 0; i < iters; i++) {
        noc.async_read(bank, dst, total_bytes, {.bank_id = src_dram_bank_id, .addr = src_dram_addr}, {});
        src_dram_addr += total_bytes;
        noc.async_read_barrier();
    }
}
