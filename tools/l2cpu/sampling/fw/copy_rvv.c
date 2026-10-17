// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* copy_rvv.c: bulk copy out of a TLB window or the uncached alias with 64-byte vector loads (built with V, like the
 * sampling library). Measured on the chip: an uncached window serves one 64 B NoC read per vle64 (0.37 ms per
 * 303,872 B row) versus ~400 cycles per scalar 8 B load (12 ms per row).
 * vl is computed in scalar code (min(remaining, vlmax)): GCC 13.2 miscompiles `i += vsetvl(n - i)`. */
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

void copy_from_window(void* dst, const void* src, size_t n) {
    uint8_t* d = dst;
    const uint8_t* s = src;
    if ((((uintptr_t)d | (uintptr_t)s) & 7) == 0) {
        size_t words = n / 8;
        size_t vlmax = __riscv_vsetvlmax_e64m1();
        size_t i = 0;
        while (i < words) {
            size_t rem = words - i;
            size_t vl = rem < vlmax ? rem : vlmax;
            vl = __riscv_vsetvl_e64m1(vl);
            vuint64m1_t v = __riscv_vle64_v_u64m1((const uint64_t*)s + i, vl);
            __riscv_vse64_v_u64m1((uint64_t*)d + i, v, vl);
            i += vl;
        }
        d += words * 8;
        s += words * 8;
        n -= words * 8;
    }
    while (n--) {
        *d++ = *s++;
    }
}
