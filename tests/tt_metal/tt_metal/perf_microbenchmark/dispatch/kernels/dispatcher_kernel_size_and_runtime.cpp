// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #include "debug/dprint.h"
#include "c_tensix_core.h"

#if KERNEL_SIZE_BYTES > 16
constexpr uint32_t empty_kernel_bytes = 16;
uint8_t data1[KERNEL_SIZE_BYTES - empty_kernel_bytes] __attribute__ ((section ("l1_data_test_only"))) __attribute__((used));
#endif

void kernel_main() {
    const uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUNTIME_SECONDS;
    // DPRINT << KERNEL_RUNTIME_SECONDS << ENDL();
    // DPRINT << end_time << ENDL();
    while (c_tensix_core::read_wall_clock() < end_time) {
        // DPRINT << c_tensix_core::read_wall_clock() << ENDL();
    }
}
