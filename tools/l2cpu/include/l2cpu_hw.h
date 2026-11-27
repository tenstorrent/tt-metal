// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * l2cpu_hw.h: every hardware address used by the L2CPU component, in one place.
 *
 * Two sections:
 *   L2CPU_BH_*    Blackhole L2CPU tile 0 (CPUs 0-3, NoC0 (8,3)). Measured status of each value: ../README.md
 *                 "Hardware facts".
 *   L2CPU_QEMU_*  the QEMU `virt` machine used to test the firmware logic.
 *
 * Usable from C and C++ (firmware, Tensix kernels, host-side generators). Plain integer macros only.
 * Every address the firmware and Tensix kernels use is defined here (the host package has its own copy in
 * host/l2cpu/hw.py).
 */
#ifndef L2CPU_HW_H
#define L2CPU_HW_H

/* ------------------------------------------------------------------------------------------------
 * Blackhole, L2CPU tile 0 (x280 physical addresses unless noted)
 * ---------------------------------------------------------------------------------------------- */

/* NoC #0 coordinates of the L2CPU tile for CPUs 0-3 (ISA L2CPUTile/README.md). */
#define L2CPU_BH_L2CPU_NOC_X 8u
#define L2CPU_BH_L2CPU_NOC_Y 3u

/* Local GDDR (D5): cached Memory Port and uncached System Port aliases (ISA MemoryMap.md). */
#define L2CPU_BH_MEMPORT_CACHED 0x400030000000ull
#define L2CPU_BH_MEMPORT_UNCACHED 0x000030000000ull
/* cached PA - this = uncached PA of the same local GDDR byte (measured: never mix both on one line) */
#define L2CPU_BH_UNCACHED_ALIAS_DELTA (L2CPU_BH_MEMPORT_CACHED - L2CPU_BH_MEMPORT_UNCACHED)

/* CLINT (SiFive clint0 layout) */
#define L2CPU_BH_CLINT_BASE 0x02000000ull
#define L2CPU_BH_CLINT_MSIP(h) (L2CPU_BH_CLINT_BASE + 4ull * (h))
#define L2CPU_BH_CLINT_MTIMECMP(h) (L2CPU_BH_CLINT_BASE + 0x4000ull + 8ull * (h))
#define L2CPU_BH_CLINT_MTIME (L2CPU_BH_CLINT_BASE + 0xBFF8ull)
#define L2CPU_BH_MTIME_HZ 50000000u /* DT timebase-frequency 0x2faf080 (tt-bh-linux dts); DOC */

/* PLIC (SiFive plic-1.0.0 layout). M-mode context of hart h = 2h. */
#define L2CPU_BH_PLIC_BASE 0x0C000000ull
#define L2CPU_BH_PLIC_NUM_SOURCES 128u

/* MSI catcher (ISA MSICatcher.md): 16-entry FIFO, PLIC source 6 = non-empty (level). */
#define L2CPU_BH_MSI_BASE 0x20060000ull
#define L2CPU_BH_MSI_FIFO (L2CPU_BH_MSI_BASE + 0x0ull)   /* W: push, R: pop (0 when empty) */
#define L2CPU_BH_MSI_CLEAR (L2CPU_BH_MSI_BASE + 0x4ull)  /* R: clear queue */
#define L2CPU_BH_MSI_STATUS (L2CPU_BH_MSI_BASE + 0x8ull) /* R: bit0 not full, bit8 non-empty, bit9 >= 16-hwm */
#define L2CPU_BH_MSI_HWM (L2CPU_BH_MSI_BASE + 0xCull)
#define L2CPU_BH_MSI_STATUS_NONEMPTY (1u << 8)
#define L2CPU_BH_MSI_PLIC_SOURCE 6u
/* Doorbell address as seen from the NoC (Tensix notify op): tile (8,3), passthrough PA. */
#define L2CPU_BH_MSI_NOC_ADDR 0x20060000ull
/* Same register through the high alias that bypasses the x280 core complex. */
#define L2CPU_BH_MSI_NOC_ADDR_HIGH 0xFFFFF7FEFFF60000ull

/* Small TLB windows (ISA TLBWindows.md): 224 x 2 MiB. */
#define L2CPU_BH_TLB2M_CFG_BASE 0x20000000ull /* 16 B per window: u64 local_offset, u32 lo, u32 hi */
#define L2CPU_BH_TLB2M_COUNT 224u
#define L2CPU_BH_TLB2M_SHIFT 21u
#define L2CPU_BH_TLB2M_UNCACHED_BASE 0x000430000000ull
#define L2CPU_BH_TLB2M_CACHED_BASE 0x400430000000ull
/* noc_properties_lo fields */
#define L2CPU_BH_TLB_LO_X(x) ((unsigned)(x) & 0x3Fu)
#define L2CPU_BH_TLB_LO_Y(y) (((unsigned)(y) & 0x3Fu) << 6)
#define L2CPU_BH_TLB_LO_ORDERING(o) (((unsigned)(o) & 3u) << 25)
#define L2CPU_BH_TLB_LO_NOC_SEL(n) (((unsigned)(n) & 1u) << 31)

/* Per-hart reset handler address, external peripherals scratch (ISA MemoryMap.md). */
#define L2CPU_BH_RESET_VECTOR(h) (0x20010000ull + 8ull * (h))
#define L2CPU_BH_SCRATCH_BASE 0x20010100ull /* 64 B; boot record, see l2cpu_boot.h */

/* Post-release hart-side writes (ISA L2CPUTile/README.md + Caches.md). */
#define L2CPU_BH_CSR_FEATURE_DISABLE 0x7c1
#define L2CPU_BH_L2PF_BASE(h) (0x02030000ull + 0x2000ull * (h))
#define L2CPU_BH_L2PF_BASIC_CONTROL_OFF 0x0ull
#define L2CPU_BH_L2PF_USER_CONTROL_OFF 0x4ull
#define L2CPU_BH_L2PF_BASIC_CONTROL_VAL 0x15811u
#define L2CPU_BH_L2PF_USER_CONTROL_VAL 0x38c84eu
#define L2CPU_BH_CCACHE_BASE 0x02010000ull
#define L2CPU_BH_CCACHE_WAYENABLE (L2CPU_BH_CCACHE_BASE + 0x8ull)
#define L2CPU_BH_CCACHE_WAYENABLE_VAL 15u
#define L2CPU_BH_CCACHE_FLUSH64 (L2CPU_BH_CCACHE_BASE + 0x200ull)
#define L2CPU_BH_CACHE_LINE 64u

/* ARC tile registers the host uses (NoC addresses inside the ARC tile at NoC0 (8,0)). */
#define L2CPU_BH_ARC_NOC_X 8u
#define L2CPU_BH_ARC_NOC_Y 0u
#define L2CPU_BH_ARC_L2CPU_RESET 0x80030014ull
#define L2CPU_BH_L2CPU_RESET_BIT_TILE0 (1u << 4)

/* ------------------------------------------------------------------------------------------------
 * QEMU `virt` (QEMU 8.2)
 * ---------------------------------------------------------------------------------------------- */
#define L2CPU_QEMU_DRAM_BASE 0x80000000ull
#define L2CPU_QEMU_UART_BASE 0x10000000ull /* ns16550, THR +0, LSR +5 */
#define L2CPU_QEMU_TEST_FINISHER 0x00100000ull
#define L2CPU_QEMU_CLINT_BASE 0x02000000ull
#define L2CPU_QEMU_CLINT_MSIP(h) (L2CPU_QEMU_CLINT_BASE + 4ull * (h))
#define L2CPU_QEMU_CLINT_MTIMECMP(h) (L2CPU_QEMU_CLINT_BASE + 0x4000ull + 8ull * (h))
#define L2CPU_QEMU_CLINT_MTIME (L2CPU_QEMU_CLINT_BASE + 0xBFF8ull)
#define L2CPU_QEMU_MTIME_HZ 10000000u
#define L2CPU_QEMU_PLIC_BASE 0x0C000000ull
#define L2CPU_QEMU_PLIC_NUM_SOURCES 96u
/* Goldfish RTC (PLIC source 11) is used as the doorbell's level-sensitive interrupt wire:
 * the QEMU test driver raises it by setting an alarm in the past, firmware lowers it with
 * CLEAR_INTERRUPT. Same claim -> drain -> complete sequence as the MSI catcher on the chip. */
#define L2CPU_QEMU_RTC_BASE 0x00101000ull
#define L2CPU_QEMU_RTC_ALARM_LOW 0x08u
#define L2CPU_QEMU_RTC_ALARM_HIGH 0x0Cu
#define L2CPU_QEMU_RTC_IRQ_ENABLED 0x10u
#define L2CPU_QEMU_RTC_CLEAR_ALARM 0x14u
#define L2CPU_QEMU_RTC_CLEAR_INTERRUPT 0x1Cu
#define L2CPU_QEMU_DOORBELL_PLIC_SOURCE 11u
/* Fake NoC for QEMU: tile (x, y) with x < 8, y < 4 owns an 8 MiB RAM slot. */
#define L2CPU_QEMU_FAKE_NOC_BASE 0x90000000ull
#define L2CPU_QEMU_FAKE_NOC_SLOT_SHIFT 23u
#define L2CPU_QEMU_FAKE_NOC_MAX_X 8u
#define L2CPU_QEMU_FAKE_NOC_MAX_Y 4u
/* Region placement in the QEMU tests (image at region + L2CPU_OFF_FW). */
#define L2CPU_QEMU_REGION_BASE 0x80000000ull
/* Stand-in for the chip's scratch boot record (only its PMP flag is used): last page of the 1 GiB RAM. */
#define L2CPU_QEMU_BOOT_RECORD 0xBFFFF000ull

#endif /* L2CPU_HW_H */
