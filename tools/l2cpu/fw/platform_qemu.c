// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * platform_qemu.c: QEMU `virt` stand-in for the chip.
 *   - the region and other "NoC tiles" are plain RAM (fake NoC: tile (x, y) owns an 8 MiB RAM slot),
 *   - the doorbell is an emulated MSI catcher: a 16-entry RAM FIFO plus a level-sensitive PLIC line
 *     (the goldfish RTC alarm interrupt, source 11), acknowledged with the same claim -> drain ->
 *     complete sequence that platform_bh.c uses for the real MSI catcher (source 6),
 *   - IPIs and the timer use the CLINT exactly as on the chip.
 */
#include "fw.h"
#include "platform.h"
#include "plic.h"

#define DOORBELL_CTX 0u /* hart 0, M-mode */

/* Emulated MSI catcher FIFO (single producer: the test driver; single consumer: hart 0). */
static volatile uint32_t msi_head, msi_tail;
static volatile uint32_t msi_q[16];

uint32_t plat_build_flags(void) { return FW_BUILD_QEMU; }
uint64_t plat_boot_record(void) { return L2CPU_QEMU_BOOT_RECORD; }

void plat_hart_init(uint32_t hart) {
    plat_ipi_clear(hart); /* an IPI left pending by the previous image */
    (void)hart;           /* nothing else: no prefetcher / L3 ways in QEMU */
}

void plat_boot_init(void) {
    msi_head = msi_tail = 0;
    mmio_wr32(L2CPU_QEMU_RTC_BASE + L2CPU_QEMU_RTC_IRQ_ENABLED, 1);
    mmio_wr32(L2CPU_QEMU_RTC_BASE + L2CPU_QEMU_RTC_CLEAR_INTERRUPT, 1);
    /* every context of harts 0-3 (M and S): enable bits survive a chip reset with random values; then only the
     * doorbell source is enabled, for hart 0's M context */
    plic_disable_all(L2CPU_QEMU_PLIC_BASE, 2u * L2CPU_NHARTS, L2CPU_QEMU_PLIC_NUM_SOURCES);
    plic_set_priority(L2CPU_QEMU_PLIC_BASE, L2CPU_QEMU_DOORBELL_PLIC_SOURCE, 1);
    plic_enable(L2CPU_QEMU_PLIC_BASE, DOORBELL_CTX, L2CPU_QEMU_DOORBELL_PLIC_SOURCE);
    plic_set_threshold(L2CPU_QEMU_PLIC_BASE, DOORBELL_CTX, 0);
    /* a claim the previous image never completed would block the source at the gateway: complete it */
    plic_complete(L2CPU_QEMU_PLIC_BASE, DOORBELL_CTX, L2CPU_QEMU_DOORBELL_PLIC_SOURCE);
}

uint64_t plat_mtime(void) { return mmio_rd64(L2CPU_QEMU_CLINT_MTIME); }
uint32_t plat_mtime_hz(void) { return L2CPU_QEMU_MTIME_HZ; }
void plat_set_timer(uint32_t hart, uint64_t when) { mmio_wr64(L2CPU_QEMU_CLINT_MTIMECMP(hart), when); }
void plat_ipi_send(uint32_t hart) { mmio_wr32(L2CPU_QEMU_CLINT_MSIP(hart), 1); }
/* Clear, then read back: on the chip a cleared msip can otherwise fire the IPI a second time. */
void plat_ipi_clear(uint32_t hart) {
    mmio_wr32(L2CPU_QEMU_CLINT_MSIP(hart), 0);
    (void)mmio_rd32(L2CPU_QEMU_CLINT_MSIP(hart));
}

uint64_t plat_doorbell_mie(void) { return MIP_MEIP | MIP_MTIP; }

uint32_t plat_doorbell_ack(void) {
    uint32_t src = plic_claim(L2CPU_QEMU_PLIC_BASE, DOORBELL_CTX);
    uint32_t n = 0;
    if (src == 0) {
        return 0;
    }
    if (src == L2CPU_QEMU_DOORBELL_PLIC_SOURCE) {
        /* Lower the line first, then drain: a push after this point raises it again. */
        mmio_wr32(L2CPU_QEMU_RTC_BASE + L2CPU_QEMU_RTC_CLEAR_INTERRUPT, 1);
        fence();
        while (msi_tail != msi_head) {
            (void)msi_q[msi_tail % 16];
            msi_tail = msi_tail + 1;
            n++;
        }
        if (n == 0) {
            n = 1; /* line raised without a message still counts as a doorbell */
        }
    }
    plic_complete(L2CPU_QEMU_PLIC_BASE, DOORBELL_CTX, src);
    return n;
}

#if L2CPU_TEST
void plat_qemu_ring_doorbell(uint32_t value) {
    uint32_t h = msi_head;
    if (h - msi_tail < 16) { /* MSI catcher drops writes when full */
        msi_q[h % 16] = value;
        fence();
        msi_head = h + 1;
    }
    fence();
    /* alarm at time 0 (in the past) raises the RTC interrupt line immediately */
    mmio_wr32(L2CPU_QEMU_RTC_BASE + L2CPU_QEMU_RTC_ALARM_HIGH, 0);
    mmio_wr32(L2CPU_QEMU_RTC_BASE + L2CPU_QEMU_RTC_ALARM_LOW, 0);
}
#endif

static uint8_t* fake_noc(uint8_t x, uint8_t y, uint64_t addr, uint32_t len) {
    if (x >= L2CPU_QEMU_FAKE_NOC_MAX_X || y >= L2CPU_QEMU_FAKE_NOC_MAX_Y) {
        return 0;
    }
    if (addr + len > (1ull << L2CPU_QEMU_FAKE_NOC_SLOT_SHIFT) || addr + len < addr) {
        return 0;
    }
    uint64_t slot = (uint64_t)y * L2CPU_QEMU_FAKE_NOC_MAX_X + x;
    return (uint8_t*)(uintptr_t)(L2CPU_QEMU_FAKE_NOC_BASE + (slot << L2CPU_QEMU_FAKE_NOC_SLOT_SHIFT) + addr);
}

void* plat_noc_map(uint32_t slot, uint8_t x, uint8_t y, uint8_t noc, uint64_t addr, uint32_t len) {
    (void)slot;
    (void)noc;
    if (len > PLAT_MAP_MAX) {
        return 0;
    }
    return fake_noc(x, y, addr, len);
}

int plat_noc_valid(uint8_t x, uint8_t y, uint64_t addr, uint32_t len) { return fake_noc(x, y, addr, len) != 0; }

void plat_noc_read_prepare(uint32_t slot, const void* p, uint32_t len) {
    (void)slot;
    (void)p;
    (void)len;
}
void plat_noc_write_barrier(uint32_t slot) {
    (void)slot;
    fence();
}

void plat_console_putc(char c) {
    volatile uint8_t* u = (volatile uint8_t*)(uintptr_t)L2CPU_QEMU_UART_BASE;
    while ((u[5] & 0x20) == 0) {
    }
    u[0] = (uint8_t)c;
}

void plat_finish(uint32_t code) {
    mmio_wr32(L2CPU_QEMU_TEST_FINISHER, code == 0 ? 0x5555u : ((code << 16) | 0x3333u));
    for (;;) {
        wfi();
    }
}
