// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    int i = 1;
    DEVICE_PRINT("i={}\n", i);

    DEVICE_PRINT("unknown={}\n", 5);

    unsigned u = 42;
    DEVICE_PRINT("u={}\n", u);

    long long ll = -123456789012345LL;
    DEVICE_PRINT("ll={}\n", ll);

    unsigned long long ull = 123456789012345ULL;
    DEVICE_PRINT("ull={}\n", ull);

    short s = -12345;
    DEVICE_PRINT("s={}\n", s);

    unsigned short us = 12345;
    DEVICE_PRINT("us={}\n", us);

    const volatile long long unsigned int cvllu = 98765432109876ULL;
    DEVICE_PRINT("cvllu={}\n", cvllu);
}
