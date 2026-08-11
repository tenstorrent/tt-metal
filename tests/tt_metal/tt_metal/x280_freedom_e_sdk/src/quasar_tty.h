// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef QUASAR_TTY_H_
#define QUASAR_TTY_H_

#include <stdint.h>

#define QUASAR_TTY_BUF_SIZE 2048
#define QUASAR_TTY_MAGIC 0x28028028u /* "x280" flavoured, easy to spot in a dump */

// Header is deliberately first and fixed-width so a host-side reader can pick
// up magic/len without knowing anything else about the build.
struct quasar_tty_buf {
    uint32_t magic;
    uint32_t len;
    uint32_t dropped;
    char data[QUASAR_TTY_BUF_SIZE];
};

extern volatile struct quasar_tty_buf quasar_tty;

void quasar_tty_init(void);
uintptr_t quasar_tty_address(void);
uint32_t quasar_tty_length(void);

#endif  // QUASAR_TTY_H_
