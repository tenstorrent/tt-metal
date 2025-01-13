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
    uint32_t num_cols = get_arg_val<uint32_t>(2);
    uint32_t multicore = get_arg_val<uint32_t>(3);
    uint32_t test_width = get_arg_val<uint32_t>(4);
    uint32_t test_width_num = get_arg_val<uint32_t>(5);
    uint32_t num_chars = get_arg_val<uint32_t>(6);

    // Wait for the dprint raise from the brisc on the preceeding core, but not
    // for core {0, 0}.
    if (multicore > 0 && x + y != 0) {
        uint32_t wait_x = x - 1;
        uint32_t wait_y = y;
        if (x == 0) {
            // Account for wrapping
            wait_x = num_cols - 1;
            wait_y = y - 1;
        }
        // Now wait on previous core's brisc to raise a signal
        DPRINT << WAIT{wait_x + wait_y * 20 + 20000};
    }

    // Do some test printing
    // Use a string of size 17 here (non-multiple of 4) due to previous bug
    // with .rodata alignment for non-multiple of 4 strings.
    DPRINT << "TestConstCharStr";
    DPRINT << 'N' << 'C' << '{' << x << ',' << y << '}' << ENDL();
    DPRINT << SETW(test_width) << test_width_num << ENDL();
    DPRINT << SETPRECISION(4) << 0.123456f << ENDL();
    DPRINT << FIXED() << F32(0.12f) << ENDL();
    DPRINT << BF16(0x3dfb) << ENDL();  // 0.12255859375
    for (uint32_t num = 0; num < num_chars; num++) {
        DPRINT << '-';
    }
    DPRINT << ENDL();

    // Raise the signal for this core's brisc to start printing.
    DPRINT << RAISE{x * 5 + y * 1000};
}
