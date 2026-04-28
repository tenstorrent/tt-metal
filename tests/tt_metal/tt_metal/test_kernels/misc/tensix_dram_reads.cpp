// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRAM reads over noc into Tensix L1

#include "api/compile_time_args.h"
#include "experimental/core_local_mem.h"
#include "experimental/noc.h"
#include "experimental/endpoints.h"

// needs bank_id, addr to read from
// also size and a dst L1 addr
void kernel_main() {
    constexpr uint32_t src_dram_bank_id = get_compile_time_arg_val(0);
    constexpr uint32_t src_dram_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dst_l1_addr = get_compile_time_arg_val(2);
    constexpr uint32_t total_bytes = get_compile_time_arg_val(3);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> bank;
    experimental::CoreLocalMem<uint32_t> dst(dst_l1_addr);

    noc.async_read(bank, dst, total_bytes, {.bank_id = src_dram_bank_id, .addr = src_dram_addr}, {});
    noc.async_read_barrier();
}
