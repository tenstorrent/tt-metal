// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    for (int i = 0; i < 200; i++) {
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < 200; j++) {
            asm("nop");
        }
    }
}
