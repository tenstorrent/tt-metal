// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "tensix_types.h"

#include "debug/dprint.h"
#include "debug/dprint_pages.h"

// DRAM to L1 read
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t num_banks = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_bank = get_compile_time_arg_val(2);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", num_banks * pages_per_bank * page_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    uint32_t dst_addr = l1_addr;

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            dst_addr = l1_addr;
            for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
                uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);
                noc_async_read_one_packet_set_state(src_noc_addr, page_size_bytes);
                for (uint32_t i = 0; i < pages_per_bank; i++) {
                    noc_async_read_one_packet_with_state(src_noc_addr + i * page_size_bytes, dst_addr);
                    dst_addr += page_size_bytes;
                }
            }
        }
        noc_async_read_barrier();
    }
}
