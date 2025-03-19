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

// Let the LTO decide if this needs to be inline.
void l1_to_local_mem_copy(uint32_t* dst, uint32_t __attribute__((rvtt_l1_ptr))* src, int32_t len) {
#pragma GCC unroll 0
    while (len >= 3) {
        auto v0 = src[0], v1 = src[1], v2 = src[2];
        // 1) Make sure the optimizer does not think this is memcpy by
        // hiding the pointer bookkeeping in an asm.
        // 2) The scheduler doesn't know the above loads have 6 cycle
        // latency. We emit the 3 bookkeeping adds as a single block
        // in the load shadow before the stores. The optimizer will
        // not be able to move these.
        // 3) We don't need early clobbers here because of the +r
        // constraint -- early clobbers would pessimize.
        asm inline(
            "addi %0,%0,3*%3\n\t"
            "addi %1,%1,3*%3\n\t"
            "addi %2,%2,-3"
            : "+r"(src), "+r"(dst), "+r"(len)
            : "i"(sizeof(v0)));
        dst[-3] = v0, dst[-2] = v1, dst[-1] = v2;
    }
    // There are 0, 1 or 2 words of residue. This is smaller than a loop.
    // We get smaller code layout by expecting the conditions to be true.
    if (__builtin_expect(len >= 1, true)) {
        dst[0] = src[0];
        if (__builtin_expect(len >= 2, true)) {
            dst[1] = src[1];
        }
    }
}
