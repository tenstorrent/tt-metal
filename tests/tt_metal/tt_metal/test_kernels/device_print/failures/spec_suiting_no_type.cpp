// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // A spec that suits no type: the 0 flag with a width of 0, which fmt reads as the 0 flag twice
    DEVICE_PRINT("Failure: {:00x}\n", 5u);
}
