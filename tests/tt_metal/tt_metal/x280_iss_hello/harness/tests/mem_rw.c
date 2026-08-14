// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Guest-side transform: invert every preloaded byte so the host dump can
// prove both load and store through the harness.

#include <stdint.h>

#include "guest.h"
#include "htif.h"

int main(void) {
    volatile uint8_t* p = (volatile uint8_t*)HARNESS_DATA_BASE;
    const uint32_t magic = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    if (magic != HARNESS_MAGIC) {
        htif_puts("FAIL mem_rw magic\n");
        return 1;
    }
    for (unsigned i = 0; i < 256; ++i) {
        p[i] = (uint8_t)(p[i] ^ 0xFFu);
    }
    __asm__ volatile("fence ow, ow");
    htif_puts("PASS mem_rw invert\n");
    return 0;
}
