// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Hello world for a simulated Blackhole L2CPU X280 (4-hart rv64gcv, VLEN=512).
// Proves the ISS is X280-shaped: misa.V, vlenb==64, vsetvli vl==16 for e32/m1.

#include "htif.h"

#include <stdint.h>

#define X280_LIM_BASE 0x08000000UL
#define X280_ACTIVE_FW_LOAD_ADDR 0x08001000UL
#define X280_SENTINEL_ADDR ((volatile uint64_t*)0x08100000UL)
#define X280_SENTINEL_VALUE 0xDEADBEEFCAFEBABEULL
#define VEC_EXPECTED_VLENB 64UL /* VLEN=512 */
#define VEC_EXPECTED_VL 16UL    /* VLEN/SEW = 512/32, LMUL=1 */

#define READ_CSR(name)                                      \
    ({                                                      \
        uintptr_t __v;                                      \
        __asm__ __volatile__("csrr %0, " name : "=r"(__v)); \
        __v;                                                \
    })

extern char _start[];

static void line(const char* k, const char* v) {
    htif_puts("  ");
    htif_puts(k);
    htif_puts(" = ");
    htif_puts(v);
    htif_puts("\n");
}

int main(void) {
    htif_puts("\n");
    htif_puts("=====================================================================\n");
    htif_puts(" hello_x280 -- X280 ISS (rv64gcv, VLEN=512, 4 harts, LIM @ 0x08000000)\n");
    htif_puts("=====================================================================\n\n");

    const uintptr_t hartid = READ_CSR("mhartid");
    const uintptr_t misa = READ_CSR("misa");
    const int has_v = (int)((misa >> 21) & 1);

    htif_puts("[1] core identity\n");
    htif_puts("  mhartid = ");
    htif_put_u64_dec((unsigned long)hartid);
    htif_puts("\n  misa    = ");
    htif_put_u64_hex((unsigned long)misa);
    htif_puts("\n");
    line("misa.V (vector)", has_v ? "yes" : "no");

    unsigned long vlenb = 0;
    unsigned long vl = 0;
    if (has_v) {
        __asm__ __volatile__("csrr %0, vlenb" : "=r"(vlenb));
        __asm__ __volatile__("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(32));
        htif_puts("  vlenb   = ");
        htif_put_u64_dec(vlenb);
        htif_puts(" bytes -> VLEN=");
        htif_put_u64_dec(vlenb * 8);
        htif_puts("\n  vsetvli e32/m1 vl = ");
        htif_put_u64_dec(vl);
        htif_puts(" (expect 16)\n");
    }

    htif_puts("\n[2] LIM layout (tt-llm-engine x280.h)\n");
    htif_puts("  X280_LIM_BASE            = ");
    htif_put_u64_hex(X280_LIM_BASE);
    htif_puts("\n  X280_ACTIVE_FW_LOAD_ADDR = ");
    htif_put_u64_hex(X280_ACTIVE_FW_LOAD_ADDR);
    htif_puts("\n  _start                   = ");
    htif_put_u64_hex((unsigned long)(uintptr_t)_start);
    htif_puts("\n");

    volatile double a = 1.0;
    volatile double b = 3.0;
    const double q = a / b;
    (void)q;
    htif_puts("[3] hardware FP (lp64d)\n");
    htif_puts("  1.0/3.0 computed with F/D (no trap)\n\n");

    *X280_SENTINEL_ADDR = X280_SENTINEL_VALUE;
    htif_puts("[4] sentinel 0xDEADBEEFCAFEBABE written at 0x08100000\n\n");

    htif_puts("---------------------------------------------------------------------\n");
    htif_puts(" Hello, World!\n");
    htif_puts("---------------------------------------------------------------------\n\n");

    int fails = 0;
    if (hartid != 0) {
        htif_puts("FAIL: expected boot hart 0\n");
        fails++;
    }
    if (!has_v) {
        htif_puts("FAIL: misa.V is clear (not an X280-class core)\n");
        fails++;
    }
    if (vlenb != VEC_EXPECTED_VLENB) {
        htif_puts("FAIL: vlenb is not 64 (VLEN!=512)\n");
        fails++;
    }
    if (vl != VEC_EXPECTED_VL) {
        htif_puts("FAIL: vsetvli e32/m1 did not return vl=16\n");
        fails++;
    }
    if ((uintptr_t)_start != X280_ACTIVE_FW_LOAD_ADDR) {
        htif_puts("FAIL: _start is not 0x08001000\n");
        fails++;
    }
    if (*X280_SENTINEL_ADDR != X280_SENTINEL_VALUE) {
        htif_puts("FAIL: sentinel readback mismatch\n");
        fails++;
    }

    if (fails == 0) {
        htif_puts("The X280 ISS hello world ran. All checks passed.\n");
        return 0;
    }
    htif_puts("X280 ISS hello world failed.\n");
    return 1;
}
