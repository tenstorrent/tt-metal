// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// metal_tty_putc → TL1 ring buffer + tt-metal X280 cache flush (no UART on DM).
// Strong definition keeps freedom-metal tty.o out of the link.

#include <stdint.h>
#include <string.h>

#include "quasar_tty.h"
#include "x280_cache_tt.h"

// In .bss for now; on silicon pin at MEM_MAILBOX_BASE via the linker script.
volatile struct quasar_tty_buf quasar_tty __attribute__((aligned(64), used));

static void quasar_tty_flush_range(const volatile void* addr, size_t size) {
    const uintptr_t start = (uintptr_t)addr;

    // L1 D$ -> L2, one 64B line at a time (CFLUSH.D.L1 is per line).
    for (uintptr_t line = start & ~(uintptr_t)63; line < start + size; line += 64) {
        tt_x280_flush_l1_dcache(line);
    }
    // L2 -> TL1. This also probes L1 D$ for dirty lines, so it is belt and
    // braces with the loop above; both are exercised deliberately.
    tt_x280_flush_l2_cache_range(start, size);
}

void quasar_tty_init(void) {
    quasar_tty.magic = QUASAR_TTY_MAGIC;
    quasar_tty.len = 0;
    quasar_tty.dropped = 0;
    quasar_tty.data[0] = '\0';
    quasar_tty_flush_range(&quasar_tty, sizeof(quasar_tty));
}

// freedom-metal's stdio hook. gloss/sys_write.c calls this once per character,
// so newlib's printf() lands here.
int metal_tty_putc(int c) {
    if (quasar_tty.len + 1 >= QUASAR_TTY_BUF_SIZE) {
        quasar_tty.dropped++;
        return -1;
    }

    const uint32_t at = quasar_tty.len;
    quasar_tty.data[at] = (char)c;
    quasar_tty.data[at + 1] = '\0';
    quasar_tty.len = at + 1;

    // Flush on newline: one cache round trip per line rather than per byte,
    // which is the same tradeoff tt-metal's DPRINT makes.
    if (c == '\n') {
        quasar_tty_flush_range(
            &quasar_tty,
            sizeof(quasar_tty.magic) + sizeof(quasar_tty.len) + sizeof(quasar_tty.dropped) + quasar_tty.len + 1);
    }
    return c;
}

// Not used by printf, but metal/tty.h declares it and a strong definition here
// keeps tty.o out of the link for good.
int metal_tty_getc(int* c) {
    *c = -1;
    return -1;
}

uintptr_t quasar_tty_address(void) { return (uintptr_t)&quasar_tty; }

uint32_t quasar_tty_length(void) { return quasar_tty.len; }
