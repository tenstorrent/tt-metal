// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ckernel.h"
#include "api/debug/device_print.h"

void kernel_main() {
    uint32_t wait_cycles = get_arg_val<uint32_t>(0);
    uint32_t x = get_arg_val<uint32_t>(1);
    uint32_t y = get_arg_val<uint32_t>(2);
    DEVICE_PRINT("({},{}) Before wait...\n", x, y);
    ckernel::wait(wait_cycles);
    DEVICE_PRINT("({},{}) After wait...\n", x, y);
}
