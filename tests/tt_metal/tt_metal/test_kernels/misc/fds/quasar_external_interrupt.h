// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What the machine-external dispatcher kernels share. They run the firmware's own dispatcher and
// registration calls (internal/tt-2xx/machine_external_interrupts.h) from a kernel. This suite runs
// in slow dispatch, where the firmware is built without FDS and leaves interrupt table slot 11 and
// the PLIC alone, so the kernel installs the dispatcher itself and puts the slot back on the way
// out.
//
// The stimulus is the reset-state level of quasar_fds_interrupt_storm_dispatch.cpp: a group whose
// count and threshold are both zero raises its PLIC source for as long as its interrupt enable bit
// is set. No worker has to run, and every group is a separate PLIC source the kernel raises and
// lowers by itself.
//
// plic.hpp addresses PLIC context 0 only, so these kernels have to run on hart 0. A dispatch engine
// kernel does. A worker kernel runs on DM2 or above, so there is no worker-side kernel.

#include <cstdint>

#include "internal/tt-2xx/machine_external_interrupts.h"
#include "overlay/fds_signalling.hpp"
#include "quasar_fds_common.h"
#include "quasar_fds_interrupt.h"

namespace external_interrupt {

constexpr uint32_t kPlicContext = 0;
constexpr uint32_t kPriority = fds_interrupt::kFdsInterruptPriority;
constexpr uint32_t kWorkerMask = overlay::fds_signalling::all_worker_lanes_mask;
constexpr uint32_t kMieMachineExternal = uint32_t{1} << MACHINE_EXTERNAL_INTERRUPT_OFFSET;

constexpr uint32_t group_bit(uint32_t group_id) { return uint32_t{1} << group_id; }

constexpr uint32_t source_for_group(uint32_t group_id) { return fds_interrupt::fds_plic_source(group_id); }

inline uint32_t read_mie() {
    uint64_t value = 0;
    asm volatile("csrr %0, mie" : "=r"(value));
    return static_cast<uint32_t>(value);
}

inline uint32_t read_mstatus() {
    uint64_t value = 0;
    asm volatile("csrr %0, mstatus" : "=r"(value));
    return static_cast<uint32_t>(value);
}

inline bool source_enabled(uint32_t source) {
    const uint32_t enable_bits = overlay::quasar::plic_read32(
        overlay::quasar::plic_enable_register_address(source / overlay::quasar::plic_bits_per_enable_word));
    return ((enable_bits >> (source % overlay::quasar::plic_bits_per_enable_word)) & 1u) != 0;
}

// Leaves every group in group_mask at a count of zero against a threshold of zero, which is a level
// as soon as its interrupt enable bit is set. The inputs are shed first: a done captured from an
// earlier launch would hold a count above zero and the group would never fire.
inline void configure_self_raising_groups(uint32_t group_mask) {
    fds_epoch::clear_dispatch_inputs(kWorkerMask);
    for (uint32_t mask = group_mask, group = 0; mask != 0; mask >>= 1, group++) {
        if (mask & 1u) {
            overlay::FdsDispatch::fds_read_group_count(group);
            overlay::FdsDispatch::fds_config_groupid(group, kWorkerMask, 0);
        }
    }
}

// The interrupt enable register is written whole, so this raises exactly the groups named and
// lowers every other one.
inline void raise_levels(uint32_t group_mask) { overlay::FdsDispatch::fds_config_interrupt_en(group_mask); }

// Read back so the write has landed before whatever the caller does next.
inline void lower_levels() {
    overlay::FdsDispatch::fds_config_interrupt_en(0);
    overlay::FdsDispatch::fds_read_group_count(0);
}

// The body of a registered handler. The dispatcher completes the claim once the handler returns, so
// the level comes down here. It lowers every level rather than its own: a second source raised by
// the same write is already latched pending at the PLIC, and still reaches its own handler. The
// count is bumped last, so a kernel that sees it move knows the handler ran to the end.
inline void lower_levels_and_record(
    fds_kernel::status_ptr status, uint32_t count_slot, uint32_t source_slot, uint32_t plic_source) {
    const uint32_t entry = status[count_slot];
    lower_levels();
    fds_interrupt::order_fds_before_status();
    // Same guard as fds_interrupt::stop_interrupt_storm: a level this handler cannot lower would
    // re-enter it for ever, and the kernel would never run again to report it.
    if (entry >= fds_interrupt::kMaxHandlerEntries) {
        overlay::quasar::plic_disable_source(plic_source);
    }
    status[source_slot] = plic_source;
    status[count_slot] = entry + 1;
}

inline bool wait_for_slot(fds_kernel::status_ptr status, uint32_t slot, uint32_t expected, uint32_t poll_iterations) {
    for (uint32_t i = 0; i < poll_iterations; i++) {
        if (status[slot] >= expected) {
            return true;
        }
    }
    return false;
}

// True when the slot held at expected for the whole window.
inline bool slot_steady(fds_kernel::status_ptr status, uint32_t slot, uint32_t expected, uint32_t silence_iterations) {
    for (uint32_t i = 0; i < silence_iterations; i++) {
        if (status[slot] != expected) {
            return false;
        }
    }
    return true;
}

// A claim made from the kernel rather than the dispatcher, for the steps that run with mie.MEIE
// clear. Returns plic_no_source when nothing became pending.
inline uint32_t poll_claim(uint32_t poll_iterations) {
    for (uint32_t i = 0; i < poll_iterations; i++) {
        if (const uint32_t source = overlay::quasar::plic_claim()) {
            return source;
        }
    }
    return overlay::quasar::plic_no_source;
}

// Points table slot 11 at the firmware dispatcher, remembering the word it replaces. False means
// this hart is not the one PLIC context 0 belongs to, and nothing was touched. group_mask names
// every group whose source the kernel will enable, by registering it or by hand.
inline bool install_dispatcher(uint32_t group_mask, fds_interrupt::arming_state& installed) {
    if (fds_interrupt::read_mhartid() != kPlicContext) {
        return false;
    }
    installed.context = kPlicContext;
    installed.group_mask = group_mask;
    installed.saved_table_word = *fds_interrupt::interrupt_table_slot(MACHINE_EXTERNAL_INTERRUPT_OFFSET);
    init_machine_external_interrupts();
    return true;
}

inline void remove_dispatcher(const fds_interrupt::arming_state& installed) {
    // Nothing may be delivered while the sources are enabled again for the drain.
    disable_machine_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET);
    // The shared teardown sheds a pending only from a source this context has enabled, and these
    // kernels leave pendings on sources that the dispatcher or an unregister disabled.
    for (uint32_t mask = installed.group_mask, group = 0; mask != 0; mask >>= 1, group++) {
        if (mask & 1u) {
            overlay::quasar::plic_enable_source(source_for_group(group));
        }
    }
    fds_interrupt::dispatch::disarm_external_interrupt(installed);
}

}  // namespace external_interrupt
