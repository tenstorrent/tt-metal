// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    const char* s = "Hello world!";
    DEVICE_PRINT("Sample string: {}\n", s);
    DEVICE_PRINT("Compile time string: {}\n", CTSTR("Hello world!"));
    // A pointer ignores its spec, a const char* too; a CTSTR takes the specs of a string.
    DEVICE_PRINT("Sample string with a spec: {:x}\n", s);
    DEVICE_PRINT("[{:>6}]\n", CTSTR("ab"));
}
