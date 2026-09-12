// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal HTIF printf: %s %c %d %u %x %lx %lu %%

#include "iss_printf.h"

#include <stdarg.h>
#include <stdint.h>

#include "htif.h"

static void put_signed(long v) {
    if (v < 0) {
        htif_putc('-');
        htif_put_u64_dec((unsigned long)(-v));
    } else {
        htif_put_u64_dec((unsigned long)v);
    }
}

int printf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    for (const char* p = fmt; *p; p++) {
        if (*p != '%') {
            htif_putc(*p);
            continue;
        }
        p++;
        int is_long = 0;
        while (*p == 'l') {
            is_long = 1;
            p++;
        }
        switch (*p) {
            case 's': htif_puts(va_arg(ap, const char*)); break;
            case 'c': htif_putc((char)va_arg(ap, int)); break;
            case 'd': put_signed(is_long ? va_arg(ap, long) : (long)va_arg(ap, int)); break;
            case 'u':
                htif_put_u64_dec(is_long ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned));
                break;
            case 'x':
            case 'p': {
                unsigned long v = is_long ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned);
                for (int i = 7; i >= 0; --i) {
                    unsigned d = (unsigned)((v >> (i * 4)) & 0xfu);
                    htif_putc((char)(d < 10 ? '0' + d : 'a' + (d - 10)));
                }
                break;
            }
            case '%': htif_putc('%'); break;
            case '\0': va_end(ap); return 0;
            default:
                htif_putc('%');
                htif_putc(*p);
                break;
        }
    }
    va_end(ap);
    return 0;
}
