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
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);

    constexpr uint32_t transaction_size_bytes = page_size_bytes;
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", num_pages * transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    uint32_t offset = 0;
    uint32_t tile_size_bytes = page_size_bytes;

    uint32_t total_size_bytes = page_size_bytes * num_pages;
    // uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(0, src_addr);  // bank 0, offset 0, noc 0 (default)
    // noc_async_read_set_state(src_noc_addr);
    // uint32_t dram_addr = noc_async_read_tile_dram_sharded_set_state<false>(src_addr, tile_size_bytes, 0);
    {
        DeviceZoneScopedN("RISCV0");
        // do i ever need to iterate over bank ids? one set state per bank id?
        // run kernel multiple times with different bank ids?

        /*
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            for (uint32_t j = 0; j < num_pages; j++) {
                noc_async_read_with_state(src_addr + j * tile_size_bytes, l1_addr + j * tile_size_bytes,
        tile_size_bytes);
            }
        }
        noc_async_read_barrier();
        */

        for (uint32_t bank_id = 0; bank_id < 6; bank_id++) {
            // how does local l1 src and dst addr work here? and how about size
            uint64_t src_noc_addr =
                get_noc_addr_from_bank_id<true>(bank_id, src_addr);  // bank 0, offset 0, noc 0 (default)
            noc_async_read_set_state(src_noc_addr);
            for (uint32_t i = 0; i < 4; i++) {
                noc_async_read_with_state(
                    src_addr + i * tile_size_bytes, l1_addr + i * tile_size_bytes, tile_size_bytes);
            }
            // noc_async_read_with_state(src_addr, l1_addr, tile_size_bytes * 4);
            //  noc_async_read_tile_dram_sharded_with_state(dram_addr, offset, l1_addr);
            //  DPRINT << bank_id << " " << src_noc_addr << " " << src_addr << " " << l1_addr << ENDL();
            //  DPRINT << bank_id << " " << src_noc_addr << " " << offset << " " << l1_addr << ENDL();
            //  tt::data_movement::common::print_bf16_pages(l1_addr, 32 * 32, 1);
            //  src_addr += tile_size_bytes * 4;
            //  offset += tile_size_bytes;
            l1_addr += tile_size_bytes * 4;
            noc_async_read_barrier();
        }

        /*
        uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(0, src_addr); // bank 0, offset 0, noc 0 (default)
        noc_async_read_set_state(src_noc_addr);
        //how does local l1 src and dst addr work here? and how about size
        noc_async_read_with_state(l1_addr, l1_addr, total_size_bytes);
        noc_async_read_barrier();
        */
    }
}
