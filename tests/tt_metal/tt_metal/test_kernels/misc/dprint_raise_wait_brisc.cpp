
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

/*
 * A test for the WAIT and RAISE DPrint features.
 */

void kernel_main() {
    uint32_t x = get_arg_val<uint32_t>(0);
    uint32_t y = get_arg_val<uint32_t>(1);
    uint32_t multicore = get_arg_val<uint32_t>(2);
    uint32_t num_chars = get_arg_val<uint32_t>(3);

    // Wait for this core's ncrisc to raise a signal before printing to ensure ordering.
    DPRINT << WAIT{x * 5 + y * 1000};
    // Do some test printing
    DPRINT << "TestStr";
    DPRINT << 'B' << 'R' << '{' << x << ',' << y << '}' << ENDL();
    for (uint32_t num = 0; num < num_chars; num++) {
        DPRINT << '+';
    }
    DPRINT << ENDL();

    // Raise the signal for the next core's ncrisc to start printing.
    if (multicore) {
        DPRINT << RAISE{x + y * 20 + 20000};
    }
}
