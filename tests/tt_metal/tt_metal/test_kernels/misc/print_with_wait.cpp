// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ckernel.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t wait_cycles = get_arg_val<uint32_t>(0);
    uint32_t x = get_arg_val<uint32_t>(1);
    uint32_t y = get_arg_val<uint32_t>(2);
    DPRINT << "(" << x << "," << y << ") Before wait..." << ENDL();
    ckernel::wait(wait_cycles);
    DPRINT << "(" << x << "," << y << ") After wait..." << ENDL();
}
