// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    DEVICE_PRINT("Hello world!\n");
    DEVICE_PRINT(
        "First line.\n"
        "Second line.\n");
}
