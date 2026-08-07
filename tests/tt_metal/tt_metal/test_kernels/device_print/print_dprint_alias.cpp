// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() {
    DEVICE_PRINT("no args\n");
    DPRINT("no args\n");
    DEVICE_PRINT("one arg: {}\n", 42);
    DPRINT("one arg: {}\n", 42);
    DEVICE_PRINT("three args: {} {} {}\n", 1, 2, 3);
    DPRINT("three args: {} {} {}\n", 1, 2, 3);
}
