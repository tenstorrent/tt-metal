// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // Each argument is evaluated once, however many placeholders print it.
    uint32_t count = 0;
    DEVICE_PRINT("Evaluated once: {0} {0}\n", count++);
    DEVICE_PRINT("Evaluated once: {}\n", count);
}
