// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    constexpr uint32_t test_range_bytes = 0x1000;  // Max is 8K (0x2000)
    constexpr uint32_t test_range_words = test_range_bytes / sizeof(uint32_t);

    for (uint32_t loop = 0; loop < 2; ++loop) {
        for (uint32_t i = 0; i < test_range_words; ++i) {
            // Read from local memory into the temp registers
            asm volatile(
                "li t4, 0xFFB00000\n\t"  // local_mem = 0xFFB00000
                "lw x0, 0(t4)\n\t"       // local_mem[0]
                "lw x0, 16(t4)\n\t"      // local_mem[4]
                "lw x0, 32(t4)\n\t"      // local_mem[8]
                "lw x0, 48(t4)\n\t"      // local_mem[12]
                "lw x0, 64(t4)\n\t"      // local_mem[16]
                "lw x0, 80(t4)\n\t"      // local_mem[20]
                "lw x0, 96(t4)\n\t"      // local_mem[24]
                "lw x0, 112(t4)\n\t"     // local_mem[28]
                "lw x0, 128(t4)\n\t"     // local_mem[32]
                "lw x0, 144(t4)\n\t"     // local_mem[36]
                "lw x0, 160(t4)\n\t"     // local_mem[40]
                "lw x0, 176(t4)\n\t"     // local_mem[44]
                :
                :
                : "t0", "t1", "t2", "t3", "t4", "memory");
        }
    }
}
