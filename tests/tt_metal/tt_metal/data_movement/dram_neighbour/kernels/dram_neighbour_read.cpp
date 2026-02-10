#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tensix_types.h"
#include "hw/inc/api/debug/dprint.h"
// #include "impl/allocator/allocator.hpp"

// DRAM to L1 read - Neighbour variant
// Each core reads from exactly one adjacent DRAM bank (the one to its left)
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_addr = get_arg_val<uint32_t>(1);
    uint32_t bank_id = get_arg_val<uint32_t>(2);  // bank index for this core

    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_bank = get_compile_time_arg_val(1);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", pages_per_bank * page_size_bytes);
    DeviceTimestampedData("Bank id", bank_id);
    DeviceTimestampedData("Test id", test_id);

    uint32_t dst_addr = l1_addr;

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t n = 0; n < num_of_transactions; n++) {
            dst_addr = l1_addr;
            // Read only from assigned adjacent bank
            uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);
            noc_async_read_one_packet_set_state(src_noc_addr, page_size_bytes);
            for (uint32_t i = 0; i < pages_per_bank; i++) {
                noc_async_read_one_packet_with_state(src_noc_addr + i * page_size_bytes, dst_addr);
                dst_addr += page_size_bytes;
            }
        }
        noc_async_read_barrier();

    }
}
