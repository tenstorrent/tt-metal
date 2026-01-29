// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LayerNorm Reader Kernel (NCRISC/BRISC)
// Purpose: Read row-major data from DRAM into circular buffers
// - Input tensor sticks -> CB_INPUT_RM
// - Gamma tensor sticks -> CB_GAMMA_RM (once)
// - Beta tensor sticks -> CB_BETA_RM (once)
// - Generate scalar constants -> CB_SCALARS

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ==========================================================================
    // Compile-Time Args
    // ==========================================================================
    // [0] CB_INPUT_RM - circular buffer index for input row-major sticks
    // [1] CB_GAMMA_RM - circular buffer index for gamma row-major sticks
    // [2] CB_BETA_RM - circular buffer index for beta row-major sticks
    // [3] stick_size - page size for row-major data (aligned to 32 bytes)
    // [4] num_rows - total number of rows/sticks to process
    // [5..] TensorAccessorArgs for input tensor
    // [...] TensorAccessorArgs for gamma tensor
    // [...] TensorAccessorArgs for beta tensor

    constexpr uint32_t cb_input_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_beta_rm = get_compile_time_arg_val(2);
    constexpr uint32_t stick_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_rows = get_compile_time_arg_val(4);

    // Unpack TensorAccessorArgs for input tensor starting at compile-time arg offset 5
    constexpr auto input_args = TensorAccessorArgs<5>();

    // ==========================================================================
    // Runtime Args
    // ==========================================================================
    // [0] input buffer address
    // [1] gamma buffer address
    // [2] beta buffer address
    // [3] start_stick_id for this core (0 for single core)

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    // const uint32_t gamma_addr = get_arg_val<uint32_t>(1);  // Will be used in Step 2.1.2
    // const uint32_t beta_addr = get_arg_val<uint32_t>(2);   // Will be used in Step 2.1.3
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);

    // ==========================================================================
    // Step 2.1.1: Basic Input Reading
    // ==========================================================================
    // Create TensorAccessor for input tensor
    // The accessor handles address computation for interleaved DRAM layout
    const auto input_accessor = TensorAccessor(input_args, input_addr, stick_size);

    // Read all input sticks (rows) into CB_INPUT_RM
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Calculate stick ID for this row
        uint32_t stick_id = start_stick_id + row;

        // Reserve space in circular buffer for one stick
        cb_reserve_back(cb_input_rm, 1);

        // Get write pointer in L1
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        // Read stick from DRAM to L1 using tensor accessor
        // noc_async_read_page handles the address computation for interleaved layout
        noc_async_read_page(stick_id, input_accessor, l1_write_addr);

        // Wait for read to complete
        noc_async_read_barrier();

        // Push the stick to the circular buffer (make it available to compute)
        cb_push_back(cb_input_rm, 1);
    }
}
