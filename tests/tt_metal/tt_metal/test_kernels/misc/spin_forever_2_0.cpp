// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Never returns unless the firmware skips kernel_main (TT_METAL_KERNELS_EARLY_RETURN).

#include <cstdint>

void kernel_main() {
    volatile uint32_t keep_spinning = 1;
    while (keep_spinning != 0) {
    }
}
