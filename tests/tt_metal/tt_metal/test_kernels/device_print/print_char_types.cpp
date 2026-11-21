// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

void kernel_main() {
    // Plain char prints as a character. signed char and unsigned char are int8_t and uint8_t, and print as numbers.
    DEVICE_PRINT("{}\n", 'a');
    char c = 'a';
    signed char sc = 'a';
    unsigned char uc = 'a';
    DEVICE_PRINT("c={0} [{0:3}] {0:d} {0:#x} {0:c} sc={1} uc={2}\n", c, sc, uc);
}
