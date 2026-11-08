// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * lim_idle.c - Resident X280 idle / boot-handoff firmware.
 *
 * Loaded once per chip-reset into the lowest 4 KiB of LIM
 * ([X280_IDLE_FW_LOAD_ADDR, X280_IDLE_FW_LOAD_ADDR + 4 KiB)) by the
 * host loader. After the L2CPU is released from reset for the
 * first time, this firmware stays resident and each hart independently
 * handshakes with the host through its own cache-line records.
 *
 * Why: X280 cores can be released from reset only once per chip
 * reset (re-asserting reset on a running L2CPU requires a
 * chip-level reset to recover). This firmware lets the host swap
 * active firmware images without ever touching the reset bit.
 *
 * On every entry a hart publishes version, generation, and its distinct idle
 * marker before committing IDLE. It then polls only its command word and
 * jumps after observing JUMP with a stable matching generation.
 *
 * Linked with ld/x280-lim-idle.ld (X280_LIM_BASE = 0x08000000,
 * idle region size = 0x1000). Does NOT touch DRAM, AXI, or any MMIO
 * outside the local L2CPU tile -- safe to leave running across
 * Tensix / Eth init and kernel dispatch.
 */
#include "x280_address_map.h"
#include "x280_hart_protocol.h"

#include "rt/x280_hw.h"
/* Inline-asm indirect jump used to leave the idle firmware. We
 * deliberately avoid pushing a stack frame: the active FW's _start
 * re-initializes sp to its own (separate) _stack_top, so any frame
 * we leave behind here is dead. __builtin_unreachable() also tells
 * the compiler that control flow does not return, which lets it
 * elide the function epilogue entirely. */

static inline __attribute__((noreturn)) void boot_jump_to_active_fw(uint64_t entry) {
    x280_fence_ow();
    __asm__ volatile(
        /* Invalidate the L1 I-cache before jumping into the active FW.
         * The L2CPU is never reset across ping-pong handoffs, so its
         * I-cache still holds the *previous* active FW's instructions for
         * the active-FW LIM region. The host overwrote that region (over
         * NOC) with the new active FW, but NOC writes don't snoop the
         * hart's I-cache -- without this fence.i the hart would re-run the
         * stale firmware. fence.i must execute here (in the resident idle
         * FW, whose own code is never overwritten); doing it in the active
         * FW's _start would be too late, as _start itself would already be
         * fetched from the stale lines. */
        "fence.i\n"
        "jr %0\n"
        :
        : "r"(entry)
        : "memory");
    __builtin_unreachable();
}

int main(uint64_t hartid) {
    volatile x280_hart_protocol_t* protocol = (volatile x280_hart_protocol_t*)X280_HART_PROTOCOL_BASE;
    volatile x280_hart_control_t* control = &protocol->control[hartid];
    volatile x280_hart_state_t* state = &protocol->state[hartid];

    /* Re-arm writes only version, run value, heartbeat and phase. The
     * completed run's terminal_status, result and stage are deliberately left
     * intact, so cleanup()'s hart records can still be asked whether every
     * hart committed success -- X280ActiveContract asserts exactly that.
     * Zeroing the record here would fail that suite for reasons that look
     * unrelated to the image under test. */
    const uint64_t run_value = control->run_value;
    state->protocol_version = X280_HART_PROTOCOL_VERSION;
    state->run_value = run_value;
    state->heartbeat = X280_HART_IDLE_HEARTBEAT(hartid);
    x280_fence_ow();
    state->phase = X280_HART_PHASE_IDLE;

    for (;;) {
        x280_pause();
        x280_fence_ir();
        if (control->command == X280_HART_COMMAND_JUMP) {
            const uint64_t generation = control->run_value;
            const uint64_t entry = control->entry;
            if (entry == 0) {
                continue;
            }
            x280_fence_ir();
            if (control->command != X280_HART_COMMAND_JUMP || control->run_value != generation) {
                continue;
            }
            boot_jump_to_active_fw(entry);
        }
    }

    return 0; /* unreachable */
}
