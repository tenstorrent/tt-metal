// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t num_args = kernel_compile_time_args[0];
    for (uint32_t i = 1; i < num_args; i++) {
        if (kernel_compile_time_args[i] != i) {
            ASSERT(0);
            while (1);
        }
    }
}
