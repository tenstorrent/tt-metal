// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "htif.h"

#include <stdint.h>

extern volatile uint64_t tohost;

#define HTIF_DEV_CONSOLE 1ULL
#define HTIF_CMD_WRITE 1ULL

// BCD write (dev 1, cmd 1) does not respond on fromhost; Spike clears tohost
// after consuming the command.
static void htif_send(uint64_t payload) {
    while (tohost) {
    }
    tohost = payload;
    while (tohost) {
    }
}

void htif_putc(char c) { htif_send((HTIF_DEV_CONSOLE << 56) | (HTIF_CMD_WRITE << 48) | (uint8_t)c); }

void htif_puts(const char* s) {
    while (*s) {
        htif_putc(*s++);
    }
}

void htif_put_u64_hex(unsigned long v) {
    htif_puts("0x");
    for (int i = 15; i >= 0; --i) {
        unsigned d = (unsigned)((v >> (i * 4)) & 0xfu);
        htif_putc((char)(d < 10 ? '0' + d : 'a' + (d - 10)));
    }
}

void htif_put_u64_dec(unsigned long v) {
    char buf[32];
    int n = 0;
    if (v == 0) {
        htif_putc('0');
        return;
    }
    while (v && n < (int)sizeof(buf)) {
        buf[n++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while (n--) {
        htif_putc(buf[n]);
    }
}
