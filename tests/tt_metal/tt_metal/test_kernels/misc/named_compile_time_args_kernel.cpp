// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t buffer_size = get_named_compile_time_arg_val("buffer_size");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");
    constexpr uint32_t enable_debug = get_named_compile_time_arg_val("enable_debug");
    constexpr uint32_t stride_value = get_named_compile_time_arg_val("stride_value");

    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    l1_ptr[0] = buffer_size;
    l1_ptr[1] = num_tiles;
    l1_ptr[2] = enable_debug;
    l1_ptr[3] = stride_value;
}
