// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Maintain a 4096-byte buffer at a fixed L1 address: the first and last u32
// increment forever, everything between stays zero. The X280 polls the whole
// buffer through a NoC TLB window; volatile keeps every store visible.
void kernel_main() {
    constexpr uint32_t buf_addr = 0x80000;
    constexpr uint32_t buf_words = 4096 / sizeof(uint32_t);
    volatile uint32_t* buf = reinterpret_cast<volatile uint32_t*>(buf_addr);
    for (uint32_t i = 0; i < buf_words; i++) {
        buf[i] = 0;
    }
    while (true) {
        buf[0] = buf[0] + 1;
        buf[buf_words - 1] = buf[buf_words - 1] + 1;
    }
}
