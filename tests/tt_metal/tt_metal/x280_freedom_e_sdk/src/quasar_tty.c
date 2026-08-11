// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A freedom-metal console for a Quasar DM core.
//
// freedom-metal routes stdio through a single hook, metal_tty_putc(), which
// src/tty.c implements against the BSP's `stdout-path` UART -- and declares weak
// when a BSP has no UART at all. A Quasar DM core has no UART: the way anything
// leaves a DM core is by landing in Tensix L1 (TL1) and being made visible to
// the NoC, which is what tt-metal's own DPRINT does.
//
// So this file replaces the UART shim with a TL1 ring buffer, and uses
// tt-metal's X280 cache primitives to push the bytes out of the DM core's
// private L1 D$ and the shared L2 into TL1 where a reader can see them. That
// handoff -- SiFive freedom-metal stdio on top of Tenstorrent cache management
// -- is the actual integration point this demo exists to show.
//
// Providing a strong metal_tty_putc here means the linker never pulls tty.o out
// of libmetal.a, so this definition wins whether or not the BSP describes a
// UART. build.sh asserts that.

#include <stdint.h>
#include <string.h>

#include "quasar_tty.h"
#include "x280_cache_tt.h"

// Lives in .bss, i.e. inside the DM kernel window this program is linked into.
// On real silicon you would pin this at a mailbox address agreed with the host
// (see MEM_MAILBOX_BASE in dev_mem_map.h) via the linker script instead; the
// cache handoff below is identical either way.
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
