// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    if (kernel_compile_time_args.size() != NUM_COMPILE_TIME_ARGS) {
        ASSERT(0);
        while (1);
    }
    for (uint32_t i = 0; i < NUM_COMPILE_TIME_ARGS; i++) {
        if (kernel_compile_time_args[i] != i) {
            ASSERT(0);
            while (1);
        }
    }
}
