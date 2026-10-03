// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spike / X280 ISS HTIF console (device 1, cmd PUTCHAR).

#ifndef X280_ISS_HTIF_H_
#define X280_ISS_HTIF_H_

void htif_putc(char c);
void htif_puts(const char* s);
void htif_put_u64_hex(unsigned long v);
void htif_put_u64_dec(unsigned long v);

#endif
