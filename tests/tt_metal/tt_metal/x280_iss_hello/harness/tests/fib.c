// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "htif.h"

static unsigned long fib(unsigned n) {
    unsigned long a = 0;
    unsigned long b = 1;
    for (unsigned i = 0; i < n; ++i) {
        unsigned long t = a + b;
        a = b;
        b = t;
    }
    return a;
}

int main(void) {
    const unsigned long v = fib(20);
    htif_puts("fib(20)=");
    htif_put_u64_dec(v);
    htif_puts("\n");
    if (v != 6765UL) {
        htif_puts("FAIL fib\n");
        return 1;
    }
    htif_puts("PASS fib\n");
    return 0;
}
