// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

uint32_t factorial(uint32_t n) { return n == 0 ? 1 : n * factorial(n - 1); }

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    uint32_t x = get_arg_val<uint32_t>(0);
    DEVICE_PRINT("factorial({}) = {}\n", x, factorial(x));
}
