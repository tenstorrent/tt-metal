// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

void kernel_main() {
    // Define test values: 0, 1000, and (2^24 - 1)
    constexpr uint32_t test_values[] = {1, 1000, 16777215};
    constexpr uint32_t num_test_values = 3;

    // Validate that we can write up to 24 bits; use the get address API
    auto scratch_register_address = reinterpret_cast<volatile uint32_t*>(get_stream_scratch_register_address(0));
    for (uint32_t i = 0; i < 24; i++) {
        uint32_t test_value = 1 << i;
        *scratch_register_address = test_value;
        uint32_t readback = *scratch_register_address;
        if (readback != test_value) {
            ASSERT(false);
            while (1);
        }
    }
    {
        constexpr uint32_t test_value = 0xc0ffee;
        *scratch_register_address = test_value;
        uint32_t readback = read_stream_scratch_register(0);
        if (readback != test_value) {
            ASSERT(false);
            while (1);
        }

        write_stream_scratch_register(0, 1);
        readback = *scratch_register_address;
        if (readback != 1) {
            ASSERT(false);
            while (1);
        }
    }

    // Check for all stream IDs
    for (uint8_t stream_id = 0; stream_id <= 31; stream_id++) {
        // PART 1: Direct memory access test
        // Test write_stream_scratch_register() and read_stream_scratch_register()
        for (uint32_t i = 0; i < num_test_values; i++) {
            uint32_t val = test_values[i];
            write_stream_scratch_register(stream_id, val);
            uint32_t readback = read_stream_scratch_register(stream_id);
            if (readback != val) {
                ASSERT(false);
                while (1);
            }
        }

        // PART 2: NOC inline write test
        // Test noc_inline_dw_write() followed by read_stream_scratch_register()

        // Get the register address for this stream
        uint32_t reg_addr = get_stream_scratch_register_address(stream_id);

        // Convert to NOC address for current core
        uint64_t noc_addr = get_noc_addr(my_x[0], my_y[0], reg_addr, noc_index);

        for (uint32_t i = 0; i < num_test_values; i++) {
            uint32_t val = test_values[i];

            // Write using NOC inline write to register space
            noc_inline_dw_write<InlineWriteDst::REG>(noc_addr, val);

            // Wait for NOC write to complete
            noc_async_write_barrier(noc_index);

            // Read back and verify
            uint32_t readback = read_stream_scratch_register(stream_id);
            if (readback != val) {
                ASSERT(false);
                while (1);
            }

            // Clear the register to make sure we aren't crossing wires with other stream registers
            noc_inline_dw_write<InlineWriteDst::REG>(noc_addr, 0);

            // Wait for NOC write to complete
            noc_async_write_barrier(noc_index);
        }
    }
}
