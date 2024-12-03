// SPDX-FileCopyrightText: © 2023, 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstddef>

using namespace std;

extern "C" int atexit(void (*f)(void)) { return 0; }

extern "C" void exit(int ec) {
    while (1) {
        asm volatile("" ::: "memory");
    }
}

extern "C" void wzerorange(uint32_t* start, uint32_t* end) {
    // manually unrolled 4 times.
    start += 4;
#pragma GCC unroll 0
    while (start <= end) {
        start[-4] = start[-3] = start[-2] = start[-1] = 0;
        // Prevent optimizer considering this loop equivalent to
        // memset (start, 0, (end - start) * sizeof (*start)) -- that's code bloat.
        asm inline("addi %0,%0,4 * %1" : "+r"(start) : "i"(sizeof(*start)));
    }
    // There are 0, 1, 2 or 3 words of residue.
    // We get better code layout expecting the conditions to be true.
    start -= 2;
    if (__builtin_expect(start <= end, true)) {
        start[-2] = start[-1] = 0;
        start += 2;
    }
    start -= 1;
    if (__builtin_expect(start <= end, true)) {
        start[-1] = 0;
    }
}
