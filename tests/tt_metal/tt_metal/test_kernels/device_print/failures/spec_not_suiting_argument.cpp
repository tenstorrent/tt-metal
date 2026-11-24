// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // A spec that doesn't suit the argument: .2f on an integer
    DEVICE_PRINT("Failure: {:.2f}\n", 5u);
}
