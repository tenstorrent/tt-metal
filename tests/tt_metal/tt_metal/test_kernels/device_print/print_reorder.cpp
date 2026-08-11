// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    uint16_t u16_1 = 16, u16_2 = 32;
    uint32_t u32_1 = 1, u32_2 = 2, u32_3 = 4, u32_4 = 8;

    DEVICE_PRINT(
        "u16_1: {} u16_2: {} u32_1: {} u32_2: {} u32_3: {} u32_4: {}\n", u16_1, u16_2, u32_1, u32_2, u32_3, u32_4);
}
