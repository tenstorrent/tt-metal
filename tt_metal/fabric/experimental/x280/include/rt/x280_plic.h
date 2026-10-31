// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_plic.h -- the PLIC and the MSI catcher, for an image that takes the
 * NOC doorbell.
 *
 * A NOC write of a non-zero u32 to X280_MSI_CATCHER_BASE at the tile's own
 * (X, Y), from the host or a Tensix kernel, pushes the value into the catcher's
 * 16-entry FIFO (a full FIFO drops it). While the FIFO is not empty the catcher
 * holds PLIC source X280_MSI_PLIC_SOURCE high. Reading the FIFO pops one value,
 * and an empty FIFO reads 0, which is why a writer never pushes 0.
 *
 * PLIC context 2h is hart h's M-mode context, 2h + 1 its S-mode context. The
 * enable bits are not reset by an ASIC reset: they come up random in every
 * context, the doorbell source included, so an image clears every enable word
 * of every context before enabling the one source it takes.
 *
 * The harness enables no interrupt (README § 3, obligation 3). An image that
 * parks in WFI on the doorbell enables mie.MEIE with mstatus.MIE left clear:
 * WFI then returns when the source is pending and no trap is taken, so the
 * entry.S trap handler is never involved. The image disables everything it
 * enabled before it returns from main.
 */
#ifndef X280_RT_PLIC_H
#define X280_RT_PLIC_H

#include <stdint.h>

#include "rt/x280_hw.h"
#include "x280_address_map.h"

#define X280_PLIC_NUM_SOURCES 128u
#define X280_PLIC_NUM_CONTEXTS 8u /* M and S for each of the four harts */
#define X280_PLIC_M_CONTEXT(hart) (2u * (hart))

#define X280_PLIC_PRIORITY_ADDR(src) (X280_PLIC_BASE + 4ull * (src))
#define X280_PLIC_ENABLE_ADDR(ctx, word) (X280_PLIC_BASE + 0x2000ull + 0x80ull * (ctx) + 4ull * (word))
#define X280_PLIC_THRESHOLD_ADDR(ctx) (X280_PLIC_BASE + 0x200000ull + 0x1000ull * (ctx))
#define X280_PLIC_CLAIM_ADDR(ctx) (X280_PLIC_BASE + 0x200004ull + 0x1000ull * (ctx))

/* MSI catcher registers (tt-isa-documentation BlackholeA0/L2CPUTile/MSICatcher.md). */
#define X280_MSI_CATCHER_FIFO_ADDR (X280_MSI_CATCHER_BASE + 0x0ull)   /* write pushes, read pops (0 when empty) */
#define X280_MSI_CATCHER_STATUS_ADDR (X280_MSI_CATCHER_BASE + 0x8ull) /* bit 8: not empty */
#define X280_MSI_CATCHER_STATUS_NONEMPTY (1u << 8)

/* CLINT timer compare, one per hart. */
#define X280_MTIMECMP_ADDR(hart) (X280_CLINT_BASE + 0x4000ull + 8ull * (hart))

#define X280_MIP_MTIP (1ull << 7)
#define X280_MIP_MEIP (1ull << 11)
#define X280_MSTATUS_MIE (1ull << 3)

static inline uint32_t x280_mmio_read32(uint64_t addr) { return *(volatile uint32_t*)(uintptr_t)addr; }

static inline void x280_mmio_write32(uint64_t addr, uint32_t value) { *(volatile uint32_t*)(uintptr_t)addr = value; }

/* Clears every enable word of every context, so no source reaches any hart. */
static inline void x280_plic_disable_all(void) {
    for (uint32_t ctx = 0; ctx < X280_PLIC_NUM_CONTEXTS; ctx++) {
        for (uint32_t word = 0; word < X280_PLIC_NUM_SOURCES / 32u; word++) {
            x280_mmio_write32(X280_PLIC_ENABLE_ADDR(ctx, word), 0);
        }
    }
    x280_fence_io();
}

/* Enables `src` alone for `ctx` at priority 1 and threshold 0, then completes a
 * claim of `src` a previous image may have left open, which would otherwise
 * hold the source at the gateway. Returns the enable word read back. */
static inline uint32_t x280_plic_enable_only(uint32_t ctx, uint32_t src) {
    x280_mmio_write32(X280_PLIC_PRIORITY_ADDR(src), 1);
    x280_mmio_write32(X280_PLIC_ENABLE_ADDR(ctx, src / 32u), 1u << (src % 32u));
    x280_mmio_write32(X280_PLIC_THRESHOLD_ADDR(ctx), 0);
    x280_mmio_write32(X280_PLIC_CLAIM_ADDR(ctx), src);
    x280_fence_io();
    return x280_mmio_read32(X280_PLIC_ENABLE_ADDR(ctx, src / 32u));
}

/* The highest-priority pending source for `ctx`, now claimed, or 0. */
static inline uint32_t x280_plic_claim(uint32_t ctx) { return x280_mmio_read32(X280_PLIC_CLAIM_ADDR(ctx)); }

static inline void x280_plic_complete(uint32_t ctx, uint32_t src) {
    x280_mmio_write32(X280_PLIC_CLAIM_ADDR(ctx), src);
    x280_fence_io();
}

static inline int x280_msi_catcher_nonempty(void) {
    return (x280_mmio_read32(X280_MSI_CATCHER_STATUS_ADDR) & X280_MSI_CATCHER_STATUS_NONEMPTY) != 0;
}

/* Pops one value; 0 means the FIFO was empty. */
static inline uint32_t x280_msi_catcher_pop(void) { return x280_mmio_read32(X280_MSI_CATCHER_FIFO_ADDR); }

static inline uint64_t x280_csr_read_mip(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mip" : "=r"(v));
    return v;
}

static inline void x280_csr_write_mie(uint64_t v) { __asm__ volatile("csrw mie, %0" ::"r"(v) : "memory"); }

static inline void x280_csr_clear_mstatus(uint64_t bits) {
    __asm__ volatile("csrc mstatus, %0" ::"r"(bits) : "memory");
}

static inline void x280_wfi(void) { __asm__ volatile("wfi" ::: "memory"); }

static inline void x280_set_mtimecmp(uint64_t hart, uint64_t when) {
    *(volatile uint64_t*)(uintptr_t)X280_MTIMECMP_ADDR(hart) = when;
    x280_fence_io();
}

#endif /* X280_RT_PLIC_H */
