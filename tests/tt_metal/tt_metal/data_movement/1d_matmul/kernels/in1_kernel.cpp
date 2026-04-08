// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "hw/inc/api/debug/dprint.h"

// DRAM to L1 read
void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t dram_bank_id = get_compile_time_arg_val(1);

    uint32_t in1_per_core_read_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_per_core_read_size_bytes = get_arg_val<uint32_t>(1);
    uint32_t in1_output_addr = get_arg_val<uint32_t>(2);

    DPRINT << "Test id: " << test_id << ENDL();
    DPRINT << "DRAM bank id: " << dram_bank_id << ENDL();
    DPRINT << "Each core will read " << in1_per_core_read_size_bytes << " bytes from DRAM" << ENDL();
    DPRINT << "Each core will read from DRAM address: " << in1_per_core_read_addr << ENDL();
    DPRINT << "Each core will write in1 read output to L1 address starting from: " << in1_output_addr << ENDL();

    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>((uint32_t)0, (uint32_t)0x493e40);

    {
        DeviceZoneScopedN("RISCV1");
        // noc_async_read(dram_noc_addr, (uint32_t)0x1b2a0, 64);
        noc_async_read_one_packet_set_state(dram_noc_addr, 64);
        noc_async_read_one_packet_with_state(dram_noc_addr, 0x1b2a0);
        noc_async_read_barrier();
        DPRINT << "L1 contents at 0x1b2a0:" << ENDL();
        uint32_t* ptr = (uint32_t*)0x1b2a0;
        for (int i = 0; i < 16; i++) {  // 64 bytes = 16 uint32_t values
            DPRINT << "addr " << (0x1b2a0 + i * 4) << ": " << ptr[i] << ENDL();
        }
    }

    DeviceTimestampedData("Test id", test_id);
}
