// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // A format that isn't a string literal
    const char* format = "Failure: {}\n";
    DEVICE_PRINT(format, 1);
}
