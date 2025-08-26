// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Test named compile time arguments using the new get_compile_time_arg_val_by_name macro
    constexpr uint32_t buffer_size = get_compile_time_arg_val_by_name("buffer_size");
    constexpr uint32_t num_tiles = get_compile_time_arg_val_by_name("num_tiles");
    constexpr uint32_t enable_debug = get_compile_time_arg_val_by_name("enable_debug");
    constexpr uint32_t stride_value = get_compile_time_arg_val_by_name("stride_value");

    // Write results to L1 memory for test verification
    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    // Write: [buffer_size, num_tiles, enable_debug, stride_value]
    l1_ptr[0] = buffer_size;
    l1_ptr[1] = num_tiles;
    l1_ptr[2] = enable_debug;
    l1_ptr[3] = stride_value;
}
