// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    uint32_t x = get_arg_val<uint32_t>(0);
    DEVICE_PRINT("Printing uint32_t from arg: {}", x);
}
