// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Write a known pattern into HARNESS_DATA_BASE so the harness can --dump it.

#include <stdint.h>

#include "guest.h"
#include "htif.h"

int main(void) {
    volatile uint8_t* p = (volatile uint8_t*)HARNESS_DATA_BASE;
    for (unsigned i = 0; i < 256; ++i) {
        p[i] = (uint8_t)(0xA5 ^ (uint8_t)i);
    }
    __asm__ volatile("fence ow, ow");
    htif_puts("filled 256 bytes at ");
    htif_put_u64_hex(HARNESS_DATA_BASE);
    htif_puts("\nPASS fill_and_dump\n");
    return 0;
}
