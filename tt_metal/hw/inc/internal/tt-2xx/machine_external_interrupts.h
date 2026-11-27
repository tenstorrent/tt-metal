// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Every PLIC source reaches the hart through the one machine-external interrupt, mcause 11. Its vectored table slot
// holds a dispatcher that claims each pending source and calls the handler registered for it, so any number of
// sources can share the line. Interrupts with their own mcause, such as ROCC, keep their own table slots.
//
// The handler table lives in firmware, so only firmware registers handlers.

#include <cstdint>

#include "internal/firmware_common.h"
#include "api/debug/assert.h"
#include "quasar/plic.hpp"

// Called with the claimed source. A plain function: the dispatcher saves the context and returns with mret.
// The dispatcher completes the claim once this returns, so a level-triggered source must have lowered its level
// first, or the PLIC delivers it again at once.
using machine_external_interrupt_handler_t = void (*)(uint32_t plic_source);

// Written in thread context and read by the dispatcher.
inline volatile machine_external_interrupt_handler_t
    machine_external_interrupt_handlers[overlay::quasar::plic_num_sources];

__attribute__((interrupt)) inline void machine_external_interrupt_dispatcher() {
    while (const uint32_t source = overlay::quasar::plic_claim()) {
        const machine_external_interrupt_handler_t handler =
            source < overlay::quasar::plic_num_sources ? machine_external_interrupt_handlers[source] : nullptr;
        if (handler != nullptr) {
            handler(source);
            overlay::quasar::plic_complete(source);
            continue;
        }
        // The PLIC drops a completion from a context that does not have the source enabled, and a claim left in
        // flight blocks that source for good, so complete before disabling.
        overlay::quasar::plic_complete(source);
        if (source < overlay::quasar::plic_num_sources) {
            overlay::quasar::plic_disable_source(source);
        }
        ASSERT(0);
    }
}

// Takes over table slot 11 and leaves every PLIC source disabled. Call before registering any handler; mie.MEIE is
// left alone until arm_machine_external_interrupts.
inline void init_machine_external_interrupts() {
    for (uint32_t source = 0; source < overlay::quasar::plic_num_sources; ++source) {
        machine_external_interrupt_handlers[source] = nullptr;
    }
    overlay::quasar::plic_set_threshold(overlay::quasar::plic_threshold_allow_all);
    overlay::quasar::plic_disable_all_sources();
    register_handler_for_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET, machine_external_interrupt_dispatcher);
    // The table entry is a jump instruction, so the icache must drop the old one before the first interrupt.
    invalidate_l1_icache();
}

// A source is delivered only while its priority is above the threshold, so priority must be nonzero.
inline void register_machine_external_interrupt_handler(
    uint32_t plic_source, uint32_t priority, machine_external_interrupt_handler_t handler) {
    ASSERT(plic_source != overlay::quasar::plic_no_source && plic_source < overlay::quasar::plic_num_sources);
    ASSERT(priority > overlay::quasar::plic_threshold_allow_all);
    ASSERT(handler != nullptr);
    machine_external_interrupt_handlers[plic_source] = handler;
    overlay::quasar::plic_set_priority(plic_source, priority);
    overlay::quasar::plic_enable_source(plic_source);
    // A previous run that stopped inside a handler left its claim open, which blocks the source until it is completed.
    // The PLIC drops a completion for a source that is not enabled, so this follows the enable.
    overlay::quasar::plic_complete(plic_source);
}

// Disabled before the handler is dropped, so a claim already taken still finds it.
inline void unregister_machine_external_interrupt_handler(uint32_t plic_source) {
    ASSERT(plic_source != overlay::quasar::plic_no_source && plic_source < overlay::quasar::plic_num_sources);
    overlay::quasar::plic_disable_source(plic_source);
    machine_external_interrupt_handlers[plic_source] = nullptr;
}

inline void arm_machine_external_interrupts() { enable_machine_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET); }
