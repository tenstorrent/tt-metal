// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "hw/inc/compile_time_args.h"
#include "hw/inc/tt-1xx/risc_common.h"

void kernel_main() {
    constexpr uint32_t cycles_addr = get_compile_time_arg_val(0);
    constexpr uint32_t src_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(3);  // Index 3, not 4!

    uint32_t end_addr = src_addr + num_bytes;

    uint64_t start = c_tensix_core::read_wall_clock();

    for (uint32_t i = 0; i < num_iterations; i++) {
        volatile uint32_t* data = reinterpret_cast<volatile uint32_t*>(src_addr);
        while ((uint32_t)data < end_addr) {
            [[maybe_unused]] volatile uint32_t word = data[0];
            data++;
        }
    }

    uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - start;
    // Write magic value to test if kernel is actually running
    ((volatile uint32_t*)cycles_addr)[0] = 0x12345678;
    ((volatile uint32_t*)cycles_addr)[1] = 0xABCDEF00;
}
