// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    uint32_t x = get_arg_val<uint32_t>(0);
    DPRINT << "Printing int from arg: " << x << ENDL();
}
