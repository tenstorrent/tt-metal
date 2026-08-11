// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// freedom-metal stdio on a Blackhole L2CPU X280.
//
// freedom-metal funnels all stdio through one hook, metal_tty_putc(), which
// src/tty.c implements against the BSP's `stdout-path` UART and leaves *weak*
// when a BSP has no UART. An L2CPU tile has no UART -- host-visible output goes
// into LIM and is read back over the NOC, which is exactly what every firmware
// in tt-llm-engine's x280/ does with its sentinels and mailboxes.
//
// So this is the whole console: printf -> newlib -> gloss/sys_write.c ->
// metal_tty_putc -> a LIM block at a fixed address the host can NOC-read.
//
// Defining metal_tty_putc strongly here means the linker never pulls tty.o out
// of libmetal.a, so this wins regardless of what the derived BSP says about
// UARTs. build.sh asserts that.

#include <stdint.h>

#include "x280_lim_console.h"

void x280_console_init(void) {
    X280_CONSOLE->magic = X280_CONSOLE_MAGIC;
    X280_CONSOLE->len = 0;
    X280_CONSOLE->dropped = 0;
    X280_CONSOLE->data[0] = '\0';
}

#ifdef X280_QEMU
// Emulation only. qemu's `sifive_u` machine has a SiFive UART at 0x10010000, so
// mirroring each byte there makes the same output visible on a terminal. A real
// L2CPU tile has no UART and this is compiled out; the LIM path below is the one
// that matters on hardware.
#define X280_QEMU_UART_TXDATA ((volatile uint32_t*)0x10010000UL)
static void qemu_uart_putc(int c) { *X280_QEMU_UART_TXDATA = (uint32_t)(c & 0xff); }
#else
static inline void qemu_uart_putc(int c) { (void)c; }
#endif

int metal_tty_putc(int c) {
    qemu_uart_putc(c);
    const uint32_t at = X280_CONSOLE->len;
    if (at + 1 >= X280_CONSOLE_CAPACITY) {
        X280_CONSOLE->dropped++;
        return -1;
    }
    X280_CONSOLE->data[at] = (char)c;
    X280_CONSOLE->data[at + 1] = '\0';
    X280_CONSOLE->len = at + 1;
    return c;
}

// Not used by printf, but metal/tty.h declares it, and a strong definition here
// keeps tty.o out of the link for good.
int metal_tty_getc(int* c) {
    *c = -1;
    return -1;
}

uint32_t x280_console_length(void) { return X280_CONSOLE->len; }
