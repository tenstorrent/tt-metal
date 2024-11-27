// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "c_tensix_core.h"
#include "core_config.h"
#include "dataflow_api.h"
#include "debug/dprint.h"

constexpr uint32_t num_unique_rt_args = get_compile_time_arg_val(0);
constexpr uint32_t num_common_rt_args = get_compile_time_arg_val(1);
constexpr uint32_t unique_rt_args_vals_offset = get_compile_time_arg_val(2);
constexpr uint32_t common_rt_args_vals_offset = get_compile_time_arg_val(3);
constexpr uint32_t num_sems = get_compile_time_arg_val(4);
constexpr uint32_t expected_sem_val = get_compile_time_arg_val(5);
constexpr uint32_t num_cbs = get_compile_time_arg_val(6);
constexpr ProgrammableCoreType sem_core_type = static_cast<ProgrammableCoreType>(get_compile_time_arg_val(7));

#if KERNEL_SIZE_BYTES > 16
constexpr uint32_t empty_kernel_bytes = 16;
[[gnu::section(".text"), gnu::used]]
static uint8_t lorem_ipsum[KERNEL_SIZE_BYTES - empty_kernel_bytes];
#endif

void kernel_main() {
    const uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUNTIME_MICROSECONDS;
    while (c_tensix_core::read_wall_clock() < end_time);

    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        const uint32_t rt_arg = get_arg_val<uint32_t>(i);
        const uint32_t expected = i + unique_rt_args_vals_offset;
        if (rt_arg != expected) {
            DPRINT << "Actual runtime argument value: " << rt_arg << " Expected runtime argument value: " << expected
                   << ENDL();
            ASSERT(0);
            while (true);  // Hang kernel if values aren't correct
        }
    }

    for (uint32_t i = num_unique_rt_args; i < num_unique_rt_args + num_sems; i++) {
        const uint32_t sem_id = get_arg_val<uint32_t>(i);
        const uint32_t actual_sem_val =
            *(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<sem_core_type>(sem_id)));
        if (expected_sem_val != actual_sem_val) {
            DPRINT << "Actual semaphore value: " << actual_sem_val << " Expected semaphore value: " << expected_sem_val
                   << ENDL();
            ASSERT(0);
            while (true);  // Hang kernel if values aren't correct
        }
    }

    uint32_t cb_idx = 0;
    for (uint32_t i = num_unique_rt_args + num_sems; i < num_unique_rt_args + num_sems + num_cbs; i++) {
        const uint32_t expected_cb_page_size = get_arg_val<uint32_t>(i);
        const uint32_t actual_cb_page_size = get_local_cb_interface(cb_idx).fifo_page_size;
        if (expected_cb_page_size != actual_cb_page_size) {
            DPRINT << cb_idx << " Actual circular buffer page size: " << actual_cb_page_size
                   << " Expected circular buffer page size: " << expected_cb_page_size << ENDL();
            ASSERT(0);
            while (true);  // Hang kernel if values aren't correct
        }
        cb_idx++;
    }

    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        const uint32_t common_rt_arg = get_common_arg_val<uint32_t>(i);
        uint32_t expected = i + common_rt_args_vals_offset;
        if (common_rt_arg != expected) {
            DPRINT << "Actual common runtime argument value: " << common_rt_arg
                   << " Expected common runtime argument value: " << expected << ENDL();
            ASSERT(0);
            FWASSERT(s, p)
            while (true);  // Hang kernel if values aren't correct
        }
    }
}
