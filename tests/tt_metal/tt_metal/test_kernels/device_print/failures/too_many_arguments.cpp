// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // All arguments must be referenced when using indexed placeholders
    DEVICE_PRINT("Unreferenced: n = {}, m = {}\n", 42, 5, 10);
}
