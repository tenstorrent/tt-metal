// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Expects the harness to --load a file at HARNESS_DATA_BASE whose first
// word is HARNESS_MAGIC and whose remaining bytes are 0x00, 0x01, ...

#include <stdint.h>

#include "guest.h"
#include "htif.h"

int main(void) {
    const volatile uint8_t* p = (volatile const uint8_t*)HARNESS_DATA_BASE;
    const uint32_t magic = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    htif_puts("loaded magic=");
    htif_put_u64_hex(magic);
    htif_puts("\n");
    if (magic != HARNESS_MAGIC) {
        htif_puts("FAIL load magic\n");
        return 1;
    }
    for (unsigned i = 4; i < 256; ++i) {
        if (p[i] != (uint8_t)i) {
            htif_puts("FAIL load byte at ");
            htif_put_u64_dec(i);
            htif_puts("\n");
            return 1;
        }
    }
    htif_puts("PASS load_and_check\n");
    return 0;
}
