// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "c_tensix_core.h"

constexpr uint32_t num_unique_rt_args = get_compile_time_arg_val(0);
constexpr uint32_t num_common_rt_args = get_compile_time_arg_val(1);
constexpr uint32_t unique_rt_args_vals_offset = get_compile_time_arg_val(2);
constexpr uint32_t common_rt_args_vals_offset = get_compile_time_arg_val(3);

#if KERNEL_SIZE_BYTES > 16
constexpr uint32_t empty_kernel_bytes = 16;
uint8_t data1[KERNEL_SIZE_BYTES - empty_kernel_bytes] __attribute__ ((section ("l1_data_test_only"))) __attribute__((used));
#endif

void kernel_main() {
    const uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUNTIME_SECONDS;
    while (c_tensix_core::read_wall_clock() < end_time);

    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        const uint32_t rt_arg = get_arg_val<uint32_t>(i);
        const uint32_t expected = i + unique_rt_args_vals_offset;
        if (rt_arg != expected) {
            while(true); // Hang kernel if values aren't correct
        }
    }

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        const uint32_t common_rt_arg = get_common_arg_val<uint32_t>(i);
        uint32_t expected = i + common_rt_args_vals_offset;
        if (common_rt_arg != expected){
            while(true); // Hang kernel if values aren't correct
        }
    }
}
