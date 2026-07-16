/*
 * x280_boot.h - Idle-FW + active-FW boot-handoff ABI for the X280 profiler.
 *
 * Ported from tenstorrent/tt-llm-engine x280/include/x280.h (handoff section),
 * with the mailbox region rebased to fit OUR LIM map (the drainer's mirrors +
 * single SPSC fill the high LIM, so their 0x08130000 region is unavailable).
 *
 * WHY: the X280 L2CPU can be released from reset only ONCE per chip reset --
 * re-asserting reset on a running L2CPU needs a chip-level reset (tt-smi -r) to
 * recover, and doing it per-run is what gives intermittent half-broken boots
 * that wedge the ARC and hang the host on the next MMIO read. So the host boots
 * a tiny resident idle FW at LIM_BASE exactly once, then ping-pongs successive
 * active FWs (the drainer) in via an indirect JUMP -- never touching reset again.
 *
 * LIM map (see also ld/x280-lim-idle.ld, ld/x280-lim.ld):
 *   [0x08000000, 0x08001000)  resident idle FW (4 KiB)                 <- reset vector target
 *   [0x08001000, 0x08010000)  active FW = drainer (profzone)           <- JUMP target
 *    0x08010000               HB counter (free SRAM, unchanged)
 *    0x08011000..0x08016000   params/results/coords + reader scratch   (unchanged)
 *    0x08016000               boot-handshake mailboxes (THIS FILE)     <- clear of scratch & MIRRORCTL
 *    0x08018000               MIRRORCTL / mirrors / single SPSC        (unchanged)
 */
#ifndef X280_BOOT_H
#define X280_BOOT_H

#include <stdint.h>

/* FW load / entry addresses. */
#define X280_IDLE_FW_LOAD_ADDR 0x08000000UL   /* reset vector points here, once */
#define X280_ACTIVE_FW_LOAD_ADDR 0x08001000UL /* drainer loaded + JUMPed to here */

/* Boot-handshake mailboxes. Each on its own 64 B cache line so the four polling
 * harts don't false-share. Sits in the free gap between the reader scratch
 * (<= 0x08016000 for nread<=4) and MIRRORCTL (0x08018000). */
#define X280_BOOT_HANDSHAKE_BASE 0x08016000UL
#define X280_IDLE_HEARTBEAT_ADDR (X280_BOOT_HANDSHAKE_BASE + 0x00UL)
#define X280_BOOT_CMD_ADDR (X280_BOOT_HANDSHAKE_BASE + 0x40UL)
#define X280_BOOT_ENTRY_ADDR_MAILBOX (X280_BOOT_HANDSHAKE_BASE + 0x80UL)
#define X280_BOOT_PHASE_ADDR (X280_BOOT_HANDSHAKE_BASE + 0xC0UL)

/* Per-hart wake mailboxes: hart 0 fans the JUMP entry out to harts 1..3 so a
 * single JUMP edge can't be consumed by one hart and missed by the others.
 * Slot 0 is unused (hart 0 polls X280_BOOT_CMD_ADDR directly) so index == mhartid. */
#define X280_BOOT_HART_WAKE_BASE (X280_BOOT_HANDSHAKE_BASE + 0x100UL)
#define X280_BOOT_HART_WAKE_ADDR(h) (X280_BOOT_HART_WAKE_BASE + (uint64_t)(h) * 0x40UL)

/* Active FW -> host / idle: heartbeat + command + phase magic. Values are
 * carried verbatim from tt-llm-engine so the semantics match its docs. */
#define X280_IDLE_HEARTBEAT_VALUE 0x1D1E0BEEFC0FFEE7ULL
#define X280_BOOT_CMD_NONE 0x0000000000000000ULL
#define X280_BOOT_CMD_JUMP 0x000000004A554D50ULL /* "JUMP" */
#define X280_BOOT_PHASE_IDLE 0x000000001D1E0001ULL
#define X280_BOOT_PHASE_RUNNING_ACTIVE_FW 0x000000007E570001ULL
#define X280_BOOT_PHASE_RETURNED_TO_IDLE 0x000000001D1E0002ULL

#endif /* X280_BOOT_H */
