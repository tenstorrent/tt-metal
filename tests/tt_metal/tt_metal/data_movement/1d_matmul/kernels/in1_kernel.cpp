// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "hw/inc/api/debug/dprint.h"
#include "barrier_sync.hpp"

// DRAM to L1 read
void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t dram_bank_id = get_compile_time_arg_val(1);

    uint32_t in1_per_core_read_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_per_core_read_size_bytes = get_arg_val<uint32_t>(1);
    uint32_t in1_output_addr = get_arg_val<uint32_t>(2);
    // Barrier synchronization args
    uint32_t barrier_sem_id = get_arg_val<uint32_t>(3);
    uint32_t barrier_coord_x = get_arg_val<uint32_t>(4);
    uint32_t barrier_coord_y = get_arg_val<uint32_t>(5);
    uint32_t num_cores = get_arg_val<uint32_t>(6);
    uint32_t local_barrier_addr = get_arg_val<uint32_t>(7);

    barrier_sync(barrier_sem_id, barrier_coord_x, barrier_coord_y, num_cores, local_barrier_addr);

    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_per_core_read_addr);

    {
        DeviceZoneScopedN("RISCV1");
        // Address, destination, and size are host-validated runtime args
        noc_async_read(dram_noc_addr, in1_output_addr, in1_per_core_read_size_bytes);
        noc_async_read_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
}
