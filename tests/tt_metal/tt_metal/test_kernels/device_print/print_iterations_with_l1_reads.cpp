// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

/*
 * Prints one message per iteration, like print_iterations.cpp, and between prints reads an L1 buffer
 * through an ordinary pointer, the way a kernel reads its input data. Used to check that printing
 * stays correct while the kernel also works on L1 data.
 *
 * Runtime varargs: [0] iteration count, [1] L1 buffer address, [2] buffer size in bytes.
 */
void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    const uint32_t buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t buffer_bytes = get_arg_val<uint32_t>(2);
    const volatile uint32_t* buffer = reinterpret_cast<const volatile uint32_t*>(static_cast<uintptr_t>(buffer_addr));

    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; i++) {
        DEVICE_PRINT("Test iteration: {}\n", i);
        for (uint32_t offset = 0; offset < buffer_bytes; offset += 64) {
            sum += buffer[offset / sizeof(uint32_t)];
        }
    }
    if (sum == 0xFFFFFFFFu) {
        DEVICE_PRINT("sum: {}\n", sum);  // keeps the reads
    }
}
