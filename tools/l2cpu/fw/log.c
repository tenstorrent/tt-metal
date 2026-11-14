// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* log.c: log ring in the region + a tiny printf (%d %i %u %x %X %p %s %c %%, l/ll/z, width, 0-pad). */
#include <stdarg.h>

#include "fw.h"
#include "platform.h"

static volatile uint32_t log_lock;

int spin_trylock(volatile uint32_t* l, uint32_t tries) {
    while (tries--) {
        if (!__atomic_exchange_n(l, 1u, __ATOMIC_ACQUIRE)) {
            return 1;
        }
    }
    return 0;
}
void spin_unlock(volatile uint32_t* l) { __atomic_store_n(l, 0u, __ATOMIC_RELEASE); }

void log_init(void) {
    l2cpu_log_hdr_t* h = (l2cpu_log_hdr_t*)(g_region + L2CPU_OFF_LOG);
    if (rd32(&h->size) != L2CPU_LOG_DATA_SIZE) {
        wr64(&h->wr, 0);
        wr32(&h->size, L2CPU_LOG_DATA_SIZE);
    }
    fence();
}

static void out_ch(char* buf, int size, int* n, char c) {
    if (*n < size - 1) {
        buf[*n] = c;
    }
    (*n)++;
}

static void out_num(char* buf, int size, int* n, uint64_t v, unsigned base, int neg, int width, int zero, int upper) {
    char tmp[24];
    int k = 0;
    const char* dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    do {
        tmp[k++] = dig[v % base];
        v /= base;
    } while (v);
    int len = k + (neg ? 1 : 0);
    if (neg && zero) {
        out_ch(buf, size, n, '-');
    }
    for (; len < width; width--) {
        out_ch(buf, size, n, zero ? '0' : ' ');
    }
    if (neg && !zero) {
        out_ch(buf, size, n, '-');
    }
    while (k) {
        out_ch(buf, size, n, tmp[--k]);
    }
}

int fw_vsnprintf(char* buf, int size, const char* fmt, va_list ap) {
    int n = 0;
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            out_ch(buf, size, &n, *fmt);
            continue;
        }
        fmt++;
        int zero = 0, width = 0, lng = 0, left = 0;
        while (*fmt == '0' || *fmt == '-') {
            if (*fmt == '0') {
                zero = 1;
            } else {
                left = 1;
            }
            fmt++;
        }
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt++ - '0');
        }
        while (*fmt == 'l' || *fmt == 'z') {
            lng++;
            fmt++;
        }
        switch (*fmt) {
            case 'd':
            case 'i': {
                int64_t v = lng ? va_arg(ap, int64_t) : (int64_t)va_arg(ap, int);
                out_num(buf, size, &n, v < 0 ? (uint64_t)(-v) : (uint64_t)v, 10, v < 0, width, zero, 0);
                break;
            }
            case 'u': {
                uint64_t v = lng ? va_arg(ap, uint64_t) : (uint64_t)va_arg(ap, unsigned);
                out_num(buf, size, &n, v, 10, 0, width, zero, 0);
                break;
            }
            case 'x':
            case 'X': {
                uint64_t v = lng ? va_arg(ap, uint64_t) : (uint64_t)va_arg(ap, unsigned);
                out_num(buf, size, &n, v, 16, 0, width, zero, *fmt == 'X');
                break;
            }
            case 'p': {
                uint64_t v = (uint64_t)(uintptr_t)va_arg(ap, void*);
                out_ch(buf, size, &n, '0');
                out_ch(buf, size, &n, 'x');
                out_num(buf, size, &n, v, 16, 0, width, 1, 0);
                break;
            }
            case 's': {
                const char* s = va_arg(ap, const char*);
                if (!s) {
                    s = "(null)";
                }
                int len = (int)strlen(s);
                if (!left) {
                    for (; len < width; width--) {
                        out_ch(buf, size, &n, ' ');
                    }
                }
                while (*s) {
                    out_ch(buf, size, &n, *s++);
                }
                if (left) {
                    for (; len < width; width--) {
                        out_ch(buf, size, &n, ' ');
                    }
                }
                break;
            }
            case 'c': out_ch(buf, size, &n, (char)va_arg(ap, int)); break;
            case '%': out_ch(buf, size, &n, '%'); break;
            case 0: fmt--; break;
            default: out_ch(buf, size, &n, '%'); out_ch(buf, size, &n, *fmt);
        }
    }
    if (size > 0) {
        buf[n < size ? n : size - 1] = 0;
    }
    return n;
}

int fw_snprintf(char* buf, int size, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = fw_vsnprintf(buf, size, fmt, ap);
    va_end(ap);
    return n;
}

/* Append to the log ring (and mirror to the console). Caller holds the lock or gave up on it. */
static void log_append(const char* s, int len) {
    if (!g_region) {
        return;
    }
    l2cpu_log_hdr_t* h = (l2cpu_log_hdr_t*)(g_region + L2CPU_OFF_LOG);
    char* data = (char*)(g_region + L2CPU_OFF_LOG_DATA);
    uint64_t wr = rd64(&h->wr);
    for (int i = 0; i < len; i++) {
        data[(wr + (uint64_t)i) % L2CPU_LOG_DATA_SIZE] = s[i];
        plat_console_putc(s[i]);
    }
    fence(); /* data before the index */
    wr64(&h->wr, wr + (uint64_t)len);
}

void fw_log(const char* fmt, ...) {
    char buf[200];
    hart_local_t* me = self();
    int n = fw_snprintf(buf, 8, "[%u] ", me->hartid);
    va_list ap;
    va_start(ap, fmt);
    n += fw_vsnprintf(buf + n, (int)sizeof buf - n - 1, fmt, ap);
    va_end(ap);
    if (n > (int)sizeof buf - 2) {
        n = (int)sizeof buf - 2;
    }
    buf[n++] = '\n';
    /* In a trap the lock may be held by this very hart: do not wait forever. */
    /* bounded: a hart parked (trap, RNMI) while holding the lock must not stop the others from logging */
    int locked = spin_trylock(&log_lock, me->trap_depth ? 100000 : L2CPU_LOG_LOCK_TRIES);
    log_append(buf, n);
    if (locked) {
        spin_unlock(&log_lock);
    }
}

#if L2CPU_TEST
/* Console output for the QEMU test driver; shares the log lock so lines do not interleave. */
void console_write(const char* s) {
    int locked = spin_trylock(&log_lock, L2CPU_LOG_LOCK_TRIES);
    while (*s) {
        plat_console_putc(*s++);
    }
    if (locked) {
        spin_unlock(&log_lock);
    }
}
#endif
