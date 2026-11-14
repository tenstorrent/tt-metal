// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* libc.c: the few libc functions GCC may call. Built with -fno-builtin -fno-tree-loop-distribute-patterns. */
#include "fw.h"

void* memcpy(void* d, const void* s, size_t n) {
    uint8_t* dp = d;
    const uint8_t* sp = s;
    if ((((uintptr_t)dp | (uintptr_t)sp) & 7) == 0) {
        while (n >= 64) {
            uint64_t* d8 = (uint64_t*)dp;
            const uint64_t* s8 = (const uint64_t*)sp;
            uint64_t a = s8[0], b = s8[1], c = s8[2], e = s8[3], f = s8[4], g = s8[5], h = s8[6], i = s8[7];
            d8[0] = a;
            d8[1] = b;
            d8[2] = c;
            d8[3] = e;
            d8[4] = f;
            d8[5] = g;
            d8[6] = h;
            d8[7] = i;
            dp += 64;
            sp += 64;
            n -= 64;
        }
        while (n >= 8) {
            *(uint64_t*)dp = *(const uint64_t*)sp;
            dp += 8;
            sp += 8;
            n -= 8;
        }
    } else if ((((uintptr_t)dp | (uintptr_t)sp) & 3) == 0) {
        while (n >= 4) {
            *(uint32_t*)dp = *(const uint32_t*)sp;
            dp += 4;
            sp += 4;
            n -= 4;
        }
    } else if ((((uintptr_t)dp | (uintptr_t)sp) & 1) == 0) {
        while (n >= 2) {
            *(uint16_t*)dp = *(const uint16_t*)sp;
            dp += 2;
            sp += 2;
            n -= 2;
        }
    }
    while (n--) {
        *dp++ = *sp++;
    }
    return d;
}

void* memset(void* d, int c, size_t n) {
    uint8_t* dp = d;
    if (((uintptr_t)dp & 7) == 0) {
        uint64_t v = (uint8_t)c;
        v |= v << 8;
        v |= v << 16;
        v |= v << 32;
        while (n >= 8) {
            *(uint64_t*)dp = v;
            dp += 8;
            n -= 8;
        }
    }
    while (n--) {
        *dp++ = (uint8_t)c;
    }
    return d;
}

void* memmove(void* d, const void* s, size_t n) {
    uint8_t* dp = d;
    const uint8_t* sp = s;
    if (dp == sp || n == 0) {
        return d;
    }
    if (dp < sp || dp >= sp + n) {
        return memcpy(d, s, n);
    }
    while (n--) {
        dp[n] = sp[n];
    }
    return d;
}

int memcmp(const void* a, const void* b, size_t n) {
    const uint8_t *x = a, *y = b;
    for (size_t i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            return x[i] < y[i] ? -1 : 1;
        }
    }
    return 0;
}

size_t strlen(const char* s) {
    size_t n = 0;
    while (s[n]) {
        n++;
    }
    return n;
}
