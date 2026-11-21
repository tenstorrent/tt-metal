// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // A spec that doesn't suit a char: fmt allows 0 on a char only with an integer presentation, as in {:03d}
    DEVICE_PRINT("Failure: {:03}\n", 'a');
}
