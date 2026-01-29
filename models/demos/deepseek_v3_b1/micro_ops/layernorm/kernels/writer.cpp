// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LayerNorm Writer Kernel (BRISC/NCRISC)
// Purpose: Write row-major output from CB to DRAM
// - Wait for output sticks in CB_OUTPUT_RM
// - Write to output tensor DRAM location

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ==========================================================================
    // Compile-Time Args
    // ==========================================================================
    // [0] CB_OUTPUT_RM - output row-major CB to read from
    // [1] stick_size (page size for row-major data)
    // [2] num_rows (total rows/sticks to write)
    // [3..] TensorAccessorArgs for output tensor

    constexpr uint32_t cb_output_rm = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows = get_compile_time_arg_val(2);

    // Unpack TensorAccessorArgs for output tensor starting at compile-time arg offset 3
    constexpr auto output_args = TensorAccessorArgs<3>();

    // ==========================================================================
    // Runtime Args
    // ==========================================================================
    // [0] output buffer address
    // [1] start_stick_id for this core (0 for single core)

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);

    // ==========================================================================
    // Step 2.3.1: Basic Stick Writing (implemented for passthrough test)
    // ==========================================================================
    // Create TensorAccessor for output tensor
    const auto output_accessor = TensorAccessor(output_args, output_addr, stick_size);

    // Write all output sticks (rows) from CB_OUTPUT_RM to DRAM
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Calculate stick ID for this row
        uint32_t stick_id = start_stick_id + row;

        // Wait for output stick from compute kernel
        cb_wait_front(cb_output_rm, 1);

        // Get read pointer in L1
        uint32_t l1_read_addr = get_read_ptr(cb_output_rm);

        // Write stick from L1 to DRAM using tensor accessor
        noc_async_write_page(stick_id, output_accessor, l1_read_addr);

        // Wait for write to complete
        noc_async_write_barrier();

        // Pop the consumed stick from circular buffer
        cb_pop_front(cb_output_rm, 1);
    }
}
