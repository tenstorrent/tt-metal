// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* platform.h: what the firmware core needs from the chip (platform_bh.c) or QEMU (platform_qemu.c). */
#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>

/* NoC mapping slots. Each hart owns PLAT_SLOTS_PER_HART slots; a slot maps up to PLAT_MAP_MAX bytes
 * at any alignment. Slot ids: hart * PLAT_SLOTS_PER_HART + PLAT_SLOT_*. */
#define PLAT_SLOTS_PER_HART 4u
#define PLAT_SLOT_READ 0u
#define PLAT_SLOT_WRITE 1u
#define PLAT_SLOT_MAILBOX 2u
#define PLAT_MAP_MAX (1u << 20) /* 1 MiB per mapping request */

static inline uint32_t plat_slot(uint32_t hart, uint32_t which) { return hart * PLAT_SLOTS_PER_HART + which; }

/* Per-hart hardware setup after release (prefetcher; hart 0 also the L3 way enable). */
void plat_hart_init(uint32_t hart);
/* Hart 0, once, before READY: doorbell interrupt routing. */
void plat_boot_init(void);
uint32_t plat_build_flags(void);

uint64_t plat_mtime(void);
uint32_t plat_mtime_hz(void);
void plat_set_timer(uint32_t hart, uint64_t when);
void plat_ipi_send(uint32_t hart);
void plat_ipi_clear(uint32_t hart);

/* Doorbell (hart 0). ack: claim -> drain the message FIFO -> complete.
 * Returns the number of doorbell messages drained (0 = the claim found nothing). */
uint32_t plat_doorbell_ack(void);
/* mie bits for hart 0 in interrupt mode (doorbell + timer). */
uint64_t plat_doorbell_mie(void);

/* Map [addr, addr+len) of NoC tile (x, y) into the x280 address space through `slot`.
 * len <= PLAT_MAP_MAX. Returns NULL if the target cannot be reached. */
void* plat_noc_map(uint32_t slot, uint8_t x, uint8_t y, uint8_t noc, uint64_t addr, uint32_t len);
/* Check that a range is mappable (used by descriptor validation; must not program anything). */
int plat_noc_valid(uint8_t x, uint8_t y, uint64_t addr, uint32_t len);
/* Before a tensor read through `slot`: make sure no stale cached copy is returned. */
void plat_noc_read_prepare(uint32_t slot, const void* p, uint32_t len);
/* After tensor writes through `slot`: make them visible to NoC readers before the next publish. */
void plat_noc_write_barrier(uint32_t slot);

/* Debug console (QEMU UART; no-op on the chip) and test exit (QEMU only). */
void plat_console_putc(char c);
void plat_finish(uint32_t code) __attribute__((noreturn));

#if L2CPU_TEST
/* QEMU fake notify op: push a message into the emulated MSI FIFO and raise the doorbell line. */
void plat_qemu_ring_doorbell(uint32_t value);
#endif

#endif
