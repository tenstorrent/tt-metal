// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "htif.h"

#define READ_CSR(name)                                      \
    ({                                                      \
        unsigned long __v;                                  \
        __asm__ __volatile__("csrr %0, " name : "=r"(__v)); \
        __v;                                                \
    })

int main(void) {
    const unsigned long misa = READ_CSR("misa");
    const int has_v = (int)((misa >> 21) & 1);
    unsigned long vlenb = 0;
    unsigned long vl = 0;
    if (has_v) {
        __asm__ __volatile__("csrr %0, vlenb" : "=r"(vlenb));
        __asm__ __volatile__("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(32));
    }
    htif_puts("misa.V=");
    htif_puts(has_v ? "yes" : "no");
    htif_puts(" vlenb=");
    htif_put_u64_dec(vlenb);
    htif_puts(" vl=");
    htif_put_u64_dec(vl);
    htif_puts("\n");
    if (!has_v || vlenb != 64UL || vl != 16UL) {
        htif_puts("FAIL vector\n");
        return 1;
    }
    htif_puts("PASS vector\n");
    return 0;
}
