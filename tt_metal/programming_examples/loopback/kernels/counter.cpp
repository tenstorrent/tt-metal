// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Increment a counter at a fixed L1 address forever. The X280 polls this
// location through a NoC TLB window; volatile keeps every store visible.
void kernel_main() {
    constexpr uint32_t counter_addr = 0x80000;
    volatile uint32_t* counter = reinterpret_cast<volatile uint32_t*>(counter_addr);
    *counter = 0;
    while (true) {
        *counter = *counter + 1;
    }
}
