
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    for (int i = 0; i < 5; i ++)
    {
//Max unroll size
#pragma GCC unroll 65534
        for (int j = 0 ; j < 5; j++)
        {
            asm("nop");
        }
    }
}
