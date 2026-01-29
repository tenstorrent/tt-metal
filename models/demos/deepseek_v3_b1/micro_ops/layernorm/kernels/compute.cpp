// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LayerNorm Compute Kernel (TRISC)
// Purpose: Perform tilization, normalization, and untilization
// 1. Tilize input sticks
// 2. Tilize gamma/beta (once)
// 3. Compute mean per row
// 4. Compute variance per row
// 5. Compute rsqrt(var + epsilon)
// 6. Standardize: (x - mean) * rsqrt
// 7. Apply affine: result * gamma + beta
// 8. Untilize output

#include <cstdint>
#include "compute_kernel_api/common.h"

void kernel_main() {
    // ==========================================================================
    // Compile-Time Args
    // ==========================================================================
    // [0] CB_INPUT_RM - input row-major CB
    // [1] CB_INPUT_TILED - input tiled CB
    // [2] CB_GAMMA_RM - gamma row-major CB
    // [3] CB_GAMMA_TILED - gamma tiled CB
    // [4] CB_BETA_RM - beta row-major CB
    // [5] CB_BETA_TILED - beta tiled CB
    // [6] CB_SCALARS - scalar values CB
    // [7] CB_INTERM - intermediate results CB
    // [8] CB_OUTPUT_TILED - output tiled CB
    // [9] CB_OUTPUT_RM - output row-major CB
    // [10] tiles_per_row - number of tiles per normalization row
    // [11] num_rows - total number of rows to normalize
    // [12] W - final dimension (normalization dimension)

    constexpr uint32_t cb_input_rm = get_compile_time_arg_val(0);
    // constexpr uint32_t cb_input_tiled = get_compile_time_arg_val(1);
    // constexpr uint32_t cb_gamma_rm = get_compile_time_arg_val(2);      // Step 2.2.6
    // constexpr uint32_t cb_gamma_tiled = get_compile_time_arg_val(3);   // Step 2.2.6
    // constexpr uint32_t cb_beta_rm = get_compile_time_arg_val(4);       // Step 2.2.8
    // constexpr uint32_t cb_beta_tiled = get_compile_time_arg_val(5);    // Step 2.2.8
    // constexpr uint32_t cb_scalars = get_compile_time_arg_val(6);       // Step 2.2.2
    // constexpr uint32_t cb_interm = get_compile_time_arg_val(7);        // Step 2.2.2
    // constexpr uint32_t cb_output_tiled = get_compile_time_arg_val(8);
    constexpr uint32_t cb_output_rm = get_compile_time_arg_val(9);
    // constexpr uint32_t tiles_per_row = get_compile_time_arg_val(10);
    constexpr uint32_t num_rows = get_compile_time_arg_val(11);
    // constexpr uint32_t W = get_compile_time_arg_val(12);               // Step 2.2.2

    // ==========================================================================
    // Step 2.1.1: Minimal passthrough (NO tilize/untilize)
    // Just waits for reader to push data to CB_INPUT_RM, then signals writer
    // that data is available in CB_OUTPUT_RM.
    //
    // NOTE: This is a TEMPORARY passthrough. Full tilize/untilize implementation
    // will be in Step 2.2.1 once the reader/writer are verified working.
    //
    // For Step 2.1.1, we verify the reader can read data by:
    // - Consuming data from CB_INPUT_RM (verifies reader pushed it)
    // - Producing data in CB_OUTPUT_RM (allows writer to proceed)
    //
    // Output data will be zeros (not the input) because we can't copy memory
    // in compute kernel without using tilize/untilize. That's OK for this step.
    // ==========================================================================

    // Process each row
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Wait for input stick from reader (verifies reader is working)
        cb_wait_front(cb_input_rm, 1);

        // Reserve space in output row-major CB
        cb_reserve_back(cb_output_rm, 1);

        // Push to output (output will contain whatever was in CB_OUTPUT_RM memory)
        // In a full implementation, we would tilize, compute, untilize here.
        cb_push_back(cb_output_rm, 1);

        // Pop the consumed input stick
        cb_pop_front(cb_input_rm, 1);
    }
}
