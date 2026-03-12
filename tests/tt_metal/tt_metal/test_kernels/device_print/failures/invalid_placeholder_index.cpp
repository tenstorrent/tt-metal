// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // Placeholder index exceeds number of arguments
    DEVICE_PRINT("Bad index: n = {1}\n", 42);
}
