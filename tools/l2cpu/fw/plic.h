// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* plic.h: SiFive PLIC helpers (identical layout on Blackhole and QEMU virt). M-mode ctx of hart h = 2h. */
#ifndef PLIC_H
#define PLIC_H

#include "fw.h"

static inline void plic_set_priority(uint64_t base, uint32_t src, uint32_t prio) { mmio_wr32(base + 4ull * src, prio); }
static inline void plic_enable(uint64_t base, uint32_t ctx, uint32_t src) {
    uint64_t a = base + 0x2000ull + 0x80ull * ctx + 4ull * (src / 32);
    mmio_wr32(a, mmio_rd32(a) | (1u << (src % 32)));
}
/* Clear every enable word of contexts [0, nctx) for sources [0, nsrc). Measured on the chip: the enable registers
 * are NOT reset by a chip reset (random bits, incl. the doorbell source in worker contexts). */
static inline void plic_disable_all(uint64_t base, uint32_t nctx, uint32_t nsrc) {
    for (uint32_t ctx = 0; ctx < nctx; ctx++) {
        for (uint32_t k = 0; k < (nsrc + 31) / 32; k++) {
            mmio_wr32(base + 0x2000ull + 0x80ull * ctx + 4ull * k, 0);
        }
    }
}
static inline void plic_set_threshold(uint64_t base, uint32_t ctx, uint32_t t) {
    mmio_wr32(base + 0x200000ull + 0x1000ull * ctx, t);
}
static inline uint32_t plic_claim(uint64_t base, uint32_t ctx) {
    return mmio_rd32(base + 0x200004ull + 0x1000ull * ctx);
}
static inline void plic_complete(uint64_t base, uint32_t ctx, uint32_t src) {
    mmio_wr32(base + 0x200004ull + 0x1000ull * ctx, src);
}

#endif
