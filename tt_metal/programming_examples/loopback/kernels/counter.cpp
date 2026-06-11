// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Increment all 16 u32 of a reserved 64-byte region (one NoC flit) at a fixed L1
// address, forever. Runs on every Tensix core; the X280 polls each core's 64B
// region over the NoC. volatile keeps every store visible.
void kernel_main() {
    constexpr uint32_t buf_addr = 0x80000;
    constexpr uint32_t n = 16;  // 16 u32 = 64 bytes
    volatile uint32_t* buf = reinterpret_cast<volatile uint32_t*>(buf_addr);
    for (uint32_t i = 0; i < n; i++) {
        buf[i] = 0;
    }
    while (true) {
        for (uint32_t i = 0; i < n; i++) {
            buf[i] = buf[i] + 1;
        }
    }
}
