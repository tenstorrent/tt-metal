// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"
#include "experimental/endpoints.h"

// #include "api/debug/dprint.h"
// #include "api/debug/dprint_pages.h"

// DRAM to L1 read
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_banks = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_bank = get_compile_time_arg_val(2);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dram_bank;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", num_banks * pages_per_bank * page_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    uint32_t dst_addr = l1_addr;
    uint32_t dst_x_coord = dst_addr >> 16;
    uint32_t dst_y_coord = dst_addr & 0xFFFF;

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t n = 0; n < num_of_transactions; n++) {
            dst_addr = l1_addr;
            for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
                noc.set_async_read_state(
                    dram_bank, page_size_bytes, {.bank_id = bank_id, .addr = src_addr});
            
                for (uint32_t i = 0; i < pages_per_bank; i++) {
                    noc.async_read_with_state(
                        dram_bank,
                        unicast_endpoint,
                        page_size_bytes,
                        {.bank_id = bank_id, .addr = src_addr + i * page_size_bytes},
                        {.noc_x = dst_x_coord, .noc_y = dst_y_coord, .addr = dst_addr});
                    dst_addr += page_size_bytes;
                }
            }
        }
        noc_async_read_barrier();
    }
}
