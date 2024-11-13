// SPDX-FileCopyrightText: © 2023, 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstddef>

using namespace std;

extern "C" int atexit(void (*f)(void)) { return 0; }

extern "C" void exit(int ec) {
    while (1) { asm volatile ("" ::: "memory"); }
}

extern "C" void wzerorange(uint32_t *start, uint32_t *end) {
#pragma GCC unroll 0
    while (start != end) {
        *start = 0;
        // Prevent optimizer considering this loop equivalent to
        // memset (start, 0, end - start) -- that's code bloat.
        asm inline("addi %0,%0,%1" : "+r"(start) : "i"(sizeof(*start)));
    }
}
