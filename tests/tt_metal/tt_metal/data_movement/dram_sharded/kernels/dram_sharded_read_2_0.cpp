// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "tensix_types.h"

// DRAM to L1 read with banked AllocatorBank + transaction-ID pipelining.
void kernel_main() {
    uint32_t src_addr = get_arg(args::src_addr);
    uint32_t l1_addr = get_arg(args::l1_addr);
    uint32_t num_of_transactions = get_arg(args::num_of_transactions);
    uint32_t pages_per_bank = get_arg(args::pages_per_bank);

    constexpr uint32_t num_banks = get_arg(args::num_banks);
    constexpr uint32_t page_size_bytes = get_arg(args::page_size);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t num_of_trids = get_arg(args::num_trids);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", num_banks * pages_per_bank * page_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    Noc noc(noc_index);
    UnicastEndpoint dst_l1;
    constexpr AllocatorBankType bank_type = AllocatorBankType::DRAM;
    AllocatorBank<bank_type> src_dram;

    uint32_t dst_addr = l1_addr;
    uint32_t curr_trid = 1;  // Start trids from 1; 0 may break under fast dispatch.
    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t n = 0; n < num_of_transactions; n++) {
            dst_addr = l1_addr;
            for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
                for (uint32_t i = 0; i < pages_per_bank; i++) {
                    noc.async_read<Noc::TxnIdMode::ENABLED>(
                        src_dram,
                        dst_l1,
                        page_size_bytes,
                        {.bank_id = bank_id, .addr = src_addr + i * page_size_bytes},
                        {.addr = dst_addr},
                        NOC_UNICAST_WRITE_VC,
                        curr_trid);
                    dst_addr += page_size_bytes;
                }
                curr_trid = (curr_trid % (num_of_trids - 1)) + 1;  // keep trid in [1, num_of_trids-1]
            }
        }
        for (uint32_t t = 1; t < num_of_trids; t++) {
            noc.async_read_barrier<Noc::BarrierMode::TXN_ID>(t);
        }
    }
}
