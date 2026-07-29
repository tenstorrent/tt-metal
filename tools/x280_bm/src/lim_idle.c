/*
 * lim_idle.c - Resident X280 idle / boot-handoff firmware.
 *
 * Ported from tenstorrent/tt-llm-engine x280/src/lim_idle.c. Loaded once per
 * chip-reset into the lowest 4 KiB of LIM (X280_IDLE_FW_LOAD_ADDR) by the host.
 * After the L2CPU is released from reset the first time, this FW stays resident
 * forever; the active FW (our drainer, profzone) ping-pongs with it via the
 * boot-handoff mailboxes in x280_boot.h -- the reset bit is never touched again.
 *
 * WHY: the X280 can be released from reset only ONCE per chip reset (re-asserting
 * reset on a running L2CPU needs a chip-level reset to recover). Per-run reset
 * toggling is what gave intermittent half-broken boots that wedged the ARC and
 * hung the host on the next MMIO read. This idle FW lets the host swap active
 * firmware images without ever touching reset.
 *
 * Protocol (hart 0):
 *   1. On every _start (first boot AND each return from an active FW), publish
 *      PHASE=IDLE, CMD=NONE, zero the per-hart wake mailboxes, then stamp the
 *      heartbeat LAST (harts 1..3 gate on it).
 *   2. Busy-poll CMD for JUMP (NOT wfi -- host NOC writes raise no interrupt and
 *      wfi hard-stalls this core; a `pause` hint keeps it quiescent).
 *   3. On JUMP: latch the entry address, clear CMD, fan the entry out to harts
 *      1..3's wake mailboxes, fence.i (I-cache holds the previous active FW's
 *      bytes at that LIM region -- NOC writes don't snoop it), and jr to the entry.
 * Harts 1..3 gate on the heartbeat then poll their OWN wake mailbox (avoids a
 * multi-reader race on a single JUMP edge).
 *
 * Linked with ld/x280-lim-idle.ld (LIM_BASE = 0x08000000, 4 KiB). Shares entry.S
 * with the drainer (entry.S calls main(mhartid) on all 4 harts).
 */
#include "x280_boot.h"

/* RISC-V Zihintpause `pause` (0x0100000F). Raw word: stock binutils won't
 * assemble the mnemonic without an arch option, but the X280 implements it
 * (~30 core cycles of stall) -- keeps the spin quiescent without an interrupt. */
static inline void cpu_pause(void) { __asm__ volatile(".word 0x0100000F" ::: "memory"); }

static inline __attribute__((noreturn)) void boot_jump_to_active_fw(uint64_t entry) {
    __asm__ volatile(
        "fence ow, ow\n"
        /* Invalidate I-cache: the L2CPU is never reset across handoffs, so it
         * still holds the previous active FW's instructions for the reloaded
         * LIM region. NOC writes don't snoop the hart I-cache, so without
         * fence.i the hart would re-run stale firmware. It must execute here
         * (idle FW code is never overwritten); doing it in the active FW's
         * _start would be too late (that _start is itself fetched stale). */
        "fence.i\n"
        "jr %0\n"
        :
        : "r"(entry)
        : "memory");
    __builtin_unreachable();
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        /* Re-entrant: on first boot AND every return from an active FW, re-publish
         * the idle phase, clear the command word, and zero the per-hart wake
         * mailboxes. Wake-mailbox init must precede the heartbeat stamp because
         * harts 1..3 gate their polling on the heartbeat. */
        *(volatile uint64_t*)X280_BOOT_PHASE_ADDR = X280_BOOT_PHASE_IDLE;
        *(volatile uint64_t*)X280_BOOT_CMD_ADDR = X280_BOOT_CMD_NONE;
        for (uint64_t h = 1; h < 4; h++) {
            *(volatile uint64_t*)X280_BOOT_HART_WAKE_ADDR(h) = 0;
        }
        __asm__ volatile("fence ow, ow");
        *(volatile uint64_t*)X280_IDLE_HEARTBEAT_ADDR = X280_IDLE_HEARTBEAT_VALUE;
        __asm__ volatile("fence ow, ow");
    } else {
        /* Harts 1..3 wait until hart 0 has armed the mailboxes (heartbeat set
         * last). Guards against reading garbage LIM at chip power-on. */
        for (;;) {
            cpu_pause();
            __asm__ volatile("fence ir, ir");
            if (*(volatile uint64_t*)X280_IDLE_HEARTBEAT_ADDR == X280_IDLE_HEARTBEAT_VALUE) {
                break;
            }
        }
    }

    if (hartid == 0) {
        for (;;) {
            cpu_pause();
            __asm__ volatile("fence ir, ir");
            uint64_t cmd = *(volatile uint64_t*)X280_BOOT_CMD_ADDR;
            if (cmd == X280_BOOT_CMD_JUMP) {
                uint64_t entry = *(volatile uint64_t*)X280_BOOT_ENTRY_ADDR_MAILBOX;
                /* Ack before transferring control so the active FW sees a clean cmd. */
                *(volatile uint64_t*)X280_BOOT_CMD_ADDR = X280_BOOT_CMD_NONE;
                __asm__ volatile("fence ow, ow");
                /* Fan the entry out to harts 1..3 so they ride the same JUMP. */
                for (uint64_t h = 1; h < 4; h++) {
                    *(volatile uint64_t*)X280_BOOT_HART_WAKE_ADDR(h) = entry;
                }
                __asm__ volatile("fence ow, ow");
                boot_jump_to_active_fw(entry);
            }
        }
    } else {
        volatile uint64_t* my_wake = (volatile uint64_t*)X280_BOOT_HART_WAKE_ADDR(hartid);
        for (;;) {
            cpu_pause();
            __asm__ volatile("fence ir, ir");
            uint64_t entry = *my_wake;
            if (entry != 0) {
                /* Clear our wake before jumping so a later return-to-idle round
                 * doesn't immediately re-trigger on the same stale value. */
                *my_wake = 0;
                __asm__ volatile("fence ow, ow");
                boot_jump_to_active_fw(entry);
            }
        }
    }

    return 0; /* unreachable */
}
