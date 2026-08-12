// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// metal_tty_putc → LIM block for host NOC readback (L2CPU has no UART).
// Strong definition keeps freedom-metal tty.o out of the link.

#include <stdint.h>

#include "x280_lim_console.h"

void x280_console_init(void) {
    X280_CONSOLE->magic = X280_CONSOLE_MAGIC;
    X280_CONSOLE->len = 0;
    X280_CONSOLE->dropped = 0;
    X280_CONSOLE->data[0] = '\0';
}

#ifdef X280_QEMU
// Mirror to sifive_u UART for terminal visibility; compiled out on hardware.
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

// Declared in metal/tty.h; strong def keeps tty.o out of the link.
int metal_tty_getc(int* c) {
    *c = -1;
    return -1;
}

uint32_t x280_console_length(void) { return X280_CONSOLE->len; }
