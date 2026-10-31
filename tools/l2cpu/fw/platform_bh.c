// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * platform_bh.c: Blackhole L2CPU tile 0 (CPUs 0-3). Hardware facts used here: ../README.md "Hardware facts".
 *
 *   - Region: cached Memory Port (the image runs there; nothing to do here).
 *   - Other NoC tiles (e.g. DRAM banks): small (2 MiB) TLB windows, uncached. Each mapping slot owns two adjacent
 *     windows, so any range of up to 1 MiB at any alignment is contiguous in the x280 address space.
 *     Slot s uses windows 2s and 2s+1 (16 slots = windows 0..31; windows >= 32 are free).
 *   - Doorbell: MSI catcher -> PLIC source 6 -> hart 0 M-mode external interrupt (wfi wake only).
 */
#include "fw.h"
#include "platform.h"
#include "plic.h"

#ifndef L2CPU_BH_HWINIT
#define L2CPU_BH_HWINIT 1
#endif
#ifndef L2CPU_BH_TLB_ORDERING
#define L2CPU_BH_TLB_ORDERING 0 /* 0 default, 1 strict AXI, 2 posted, 3 counted */
#endif

#define DOORBELL_CTX 0u
#define NSLOTS (L2CPU_NHARTS * PLAT_SLOTS_PER_HART)
#define WIN_SIZE (1ull << L2CPU_BH_TLB2M_SHIFT)

typedef struct {
    uint64_t base; /* NoC address mapped at window 2s (2 MiB aligned) */
    uint32_t xy;   /* x | y << 8 | noc << 16 | valid << 24 */
    uint32_t _pad;
    volatile uint32_t* last; /* last pointer handed out (write barrier reads it back) */
} slot_state_t;

static slot_state_t slots[NSLOTS];

uint32_t plat_build_flags(void) { return 0; }
uint64_t plat_boot_record(void) { return L2CPU_BOOT_RECORD_PA; }

void plat_hart_init(uint32_t hart) {
    plat_ipi_clear(hart); /* an IPI left pending by the previous image */
#if L2CPU_BH_HWINIT
    /* CSR 0x7c1 = 0 is done in start.S. Prefetcher of this hart, recommended SiFive values. */
    mmio_wr32(L2CPU_BH_L2PF_BASE(hart) + L2CPU_BH_L2PF_BASIC_CONTROL_OFF, L2CPU_BH_L2PF_BASIC_CONTROL_VAL);
    mmio_wr32(L2CPU_BH_L2PF_BASE(hart) + L2CPU_BH_L2PF_USER_CONTROL_OFF, L2CPU_BH_L2PF_USER_CONTROL_VAL);
    if (hart == 0) {
        mmio_wr32(L2CPU_BH_CCACHE_WAYENABLE, L2CPU_BH_CCACHE_WAYENABLE_VAL); /* increase-only */
    }
    fence();
#else
    (void)hart;
#endif
}

void plat_boot_init(void) {
    /* Drain anything that arrived before we were ready (the catcher keeps values across our boot). */
    /* Bounded: a writer that never stops must not hold the boot; leftovers raise the doorbell once later. */
    for (uint32_t n = 0; n < 64 && (mmio_rd32(L2CPU_BH_MSI_STATUS) & L2CPU_BH_MSI_STATUS_NONEMPTY); n++) {
        (void)mmio_rd32(L2CPU_BH_MSI_FIFO);
    }
    /* every context of harts 0-3 (M and S): enable bits survive a chip reset with random values; then only the
     * doorbell source is enabled, for hart 0's M context */
    plic_disable_all(L2CPU_BH_PLIC_BASE, 2u * L2CPU_NHARTS, L2CPU_BH_PLIC_NUM_SOURCES);
    plic_set_priority(L2CPU_BH_PLIC_BASE, L2CPU_BH_MSI_PLIC_SOURCE, 1);
    plic_enable(L2CPU_BH_PLIC_BASE, DOORBELL_CTX, L2CPU_BH_MSI_PLIC_SOURCE);
    plic_set_threshold(L2CPU_BH_PLIC_BASE, DOORBELL_CTX, 0);
    /* a claim the previous image never completed would block the source at the gateway: complete it */
    plic_complete(L2CPU_BH_PLIC_BASE, DOORBELL_CTX, L2CPU_BH_MSI_PLIC_SOURCE);
}

uint64_t plat_mtime(void) { return mmio_rd64(L2CPU_BH_CLINT_MTIME); }
uint32_t plat_mtime_hz(void) { return L2CPU_BH_MTIME_HZ; }
void plat_set_timer(uint32_t hart, uint64_t when) { mmio_wr64(L2CPU_BH_CLINT_MTIMECMP(hart), when); }
void plat_ipi_send(uint32_t hart) { mmio_wr32(L2CPU_BH_CLINT_MSIP(hart), 1); }
/* Clear, then read back: on the chip a cleared msip can otherwise fire the IPI a second time. */
void plat_ipi_clear(uint32_t hart) {
    mmio_wr32(L2CPU_BH_CLINT_MSIP(hart), 0);
    (void)mmio_rd32(L2CPU_BH_CLINT_MSIP(hart));
}
uint64_t plat_doorbell_mie(void) { return MIP_MEIP | MIP_MTIP; }

uint32_t plat_doorbell_ack(void) {
    uint32_t src = plic_claim(L2CPU_BH_PLIC_BASE, DOORBELL_CTX);
    uint32_t n = 0;
    if (src == 0) {
        return 0;
    }
    if (src == L2CPU_BH_MSI_PLIC_SOURCE) {
        while (mmio_rd32(L2CPU_BH_MSI_STATUS) & L2CPU_BH_MSI_STATUS_NONEMPTY) {
            (void)mmio_rd32(L2CPU_BH_MSI_FIFO);
            n++;
            if (n > 64) {
                break; /* a writer faster than us; the level stays high and we come back */
            }
        }
        if (n == 0) {
            n = 1;
        }
    }
    plic_complete(L2CPU_BH_PLIC_BASE, DOORBELL_CTX, src);
    return n;
}

static void tlb_program(uint32_t win, uint8_t x, uint8_t y, uint8_t noc, uint64_t noc_addr) {
    uint64_t cfg = L2CPU_BH_TLB2M_CFG_BASE + 16ull * win;
    uint64_t off = noc_addr >> L2CPU_BH_TLB2M_SHIFT;
    uint32_t lo = L2CPU_BH_TLB_LO_X(x) | L2CPU_BH_TLB_LO_Y(y) | L2CPU_BH_TLB_LO_ORDERING(L2CPU_BH_TLB_ORDERING) |
                  L2CPU_BH_TLB_LO_NOC_SEL(noc);
    /* 32-bit accesses: the peripheral port width for 64-bit stores is not confirmed. */
    mmio_wr32(cfg + 0, (uint32_t)off);
    mmio_wr32(cfg + 4, (uint32_t)(off >> 32));
    mmio_wr32(cfg + 8, lo);
    mmio_wr32(cfg + 12, 0);
}

static uint64_t win_va(uint32_t win, int cached) {
    return (cached ? L2CPU_BH_TLB2M_CACHED_BASE : L2CPU_BH_TLB2M_UNCACHED_BASE) + (uint64_t)win * WIN_SIZE;
}

void* plat_noc_map(uint32_t slot, uint8_t x, uint8_t y, uint8_t noc, uint64_t addr, uint32_t len) {
    if (slot >= NSLOTS || len > PLAT_MAP_MAX || len == 0) {
        return 0;
    }
    slot_state_t* s = &slots[slot];
    uint32_t xy = (uint32_t)x | ((uint32_t)y << 8) | ((uint32_t)noc << 16) | (1u << 24);
    if (s->xy != xy || addr < s->base || addr + len > s->base + 2 * WIN_SIZE) {
        uint64_t base = addr & ~(WIN_SIZE - 1);
        fence(); /* finish accesses through the old mapping */
        tlb_program(2 * slot, x, y, noc, base);
        tlb_program(2 * slot + 1, x, y, noc, base + WIN_SIZE);
        fence();
        s->base = base;
        s->xy = xy;
    }
    int cached = 0;
    void* p = (void*)(uintptr_t)(win_va(2 * slot, cached) + (addr - s->base));
    s->last = (volatile uint32_t*)((uintptr_t)p & ~(uintptr_t)3);
    return p;
}

int plat_noc_valid(uint8_t x, uint8_t y, uint64_t addr, uint32_t len) {
    return x < 64 && y < 64 && len <= PLAT_MAP_MAX && addr + len >= addr;
}

void plat_noc_read_prepare(uint32_t slot, const void* p, uint32_t len) {
    (void)slot;
    (void)p;
    (void)len;
}

void plat_noc_write_barrier(uint32_t slot) {
    /* fence orders our stores; the read-back of the last written word through the same uncached
     * window makes the hart wait until the NoC write path has delivered (OPEN: A6 on the chip). */
    fence();
    if (slot < NSLOTS && slots[slot].last) {
        (void)*slots[slot].last;
    }
    fence();
}

void plat_console_putc(char c) { (void)c; }

void plat_finish(uint32_t code) {
    (void)code;
    fw_park();
}
