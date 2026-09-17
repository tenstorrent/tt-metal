// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The interrupt-delivery protocol every FDS interrupt kernel of this suite shares, in the role
// quasar_fds_epoch.h plays for the polling kernels.
//
// An FDS group raises a level, not a pulse: irq[g] is (count[g] == threshold[g]) && interrupt_en[g],
// re-evaluated continuously and never latched. The level reaches the hart as a machine external
// interrupt, mcause 11, through a tile-local PLIC in which FDS group g is source 16 + g. Three
// consequences shape everything below.
//
// Equality, not meets-or-exceeds. A count that has overshot its threshold is not a firing
// condition, and a count that has never moved is one whenever the threshold is zero too — which
// every register is out of reset. So a group is given its threshold before its interrupt enable
// bit, or arming fires with no traffic at all.
//
// Sticky at the PLIC, live at the FDS. The PLIC latches the level into a pending bit that survives
// until a context claims it, and a claim completed while the level still stands is re-delivered at
// once. Quieting a source therefore means changing what the FDS is comparing — clearing the input
// registers feeding the count, or clearing the interrupt enable bit — and doing it before the
// completion. Handlers here clear, read an FDS register back so the clear has demonstrably landed,
// and only then complete.
//
// Nothing else in tt-metal arms this path. mstatus.MIE and mie.MEIE are clear on every hart in this
// suite, and the firmware's vectored table at INTERRUPT_TABLE_BASE registers slots 0 and 13 only,
// leaving slot 11 free. That table is shared by all harts of a node and persists across launches,
// so arming saves the word it overwrites and the teardown puts it back. mie bit 13 is never
// touched: it belongs to the dataflow buffer ISR.

#include <cstdint>

#include "quasar_fds_common.h"

namespace fds_interrupt {

// The PLIC is tile-local and takes 32-bit accesses only. Priority of source s sits at +4*s; the
// per-context blocks are the enable words at +0x2000 + 0x80*c, the threshold at +0x200000 +
// 0x1000*c, and the claim/complete register one word past that threshold.
constexpr uint32_t kPlicBase = 0x08000000;
constexpr uint32_t kPlicPriorityBase = kPlicBase;
constexpr uint32_t kPlicEnableBase = kPlicBase + 0x2000;
constexpr uint32_t kPlicEnableStride = 0x80;
constexpr uint32_t kPlicThresholdBase = kPlicBase + 0x200000;
constexpr uint32_t kPlicClaimBase = kPlicBase + 0x200004;
constexpr uint32_t kPlicContextStride = 0x1000;

// FDS group g arrives at the PLIC as source 16 + g.
constexpr uint32_t kFdsPlicSourceBase = 16;

// A source is delivered only while its priority strictly exceeds the context threshold, and both
// are three-bit fields. One is the smallest value that beats the zero written below.
constexpr uint32_t kFdsInterruptPriority = 1;

// A PLIC context is a hart here, so mhartid indexes the per-context blocks directly. That the
// harts of a node are numbered 0..7 is assumed rather than confirmed, so arming refuses anything
// outside the range instead of computing an offset from a number it cannot place.
constexpr uint32_t kNumPlicContexts = 8;

constexpr uint32_t kMieMachineExternal = uint32_t{1} << MACHINE_EXTERNAL_INTERRUPT_OFFSET;
constexpr uint32_t kMstatusMie = uint32_t{1} << 3;

// The low word of mcause for a machine external interrupt. Only the low word is recorded: arriving
// through table slot 11 at all already establishes the interrupt bit, since vectored mode sends
// synchronous exceptions to slot 0.
constexpr uint32_t kMachineExternalCauseLow = MACHINE_EXTERNAL_INTERRUPT_OFFSET;

// How long a handler waits for the level it just cleared to fall before it completes the claim.
// Bounded like every wait in this suite, and generous: the level crosses a few register stages on
// its way out of the FDS block.
constexpr uint32_t kHandlerSettleIterations = 1000;

// Stale pendings a drain will shed. A handful can be latched from earlier programs; far fewer than
// this, but a source whose level still stands re-pends the instant it is completed, so the loop
// needs a bound to fail rather than spin.
constexpr uint32_t kMaxStalePendings = 64;

// A handler that completes a claim while its level still stands is re-entered immediately, so a
// level nothing here can lower becomes an unbounded storm and the kernel never runs again to report
// it. Past this many entries the handler drops the source from its context, which stops delivery
// whatever the level is doing, and the kernel then fails on the interrupt count instead of the run
// hanging. Well above the two entries the most demanding of these tests expects.
constexpr uint32_t kMaxHandlerEntries = 8;

inline uint32_t plic_read32(uint32_t address) { return *reinterpret_cast<volatile uint32_t*>(address); }

inline void plic_write32(uint32_t address, uint32_t value) { *reinterpret_cast<volatile uint32_t*>(address) = value; }

inline uint32_t read_mhartid() {
    uint64_t value = 0;
    asm volatile("csrr %0, mhartid" : "=r"(value));
    return static_cast<uint32_t>(value);
}

inline uint64_t read_mcause() {
    uint64_t value = 0;
    asm volatile("csrr %0, mcause" : "=r"(value));
    return value;
}

constexpr uint32_t fds_plic_source(uint32_t group_id) { return kFdsPlicSourceBase + group_id; }

inline uint32_t plic_claim(uint32_t context) { return plic_read32(kPlicClaimBase + context * kPlicContextStride); }

inline void plic_complete(uint32_t context, uint32_t source) {
    plic_write32(kPlicClaimBase + context * kPlicContextStride, source);
}

inline void plic_enable_source(uint32_t context, uint32_t source, bool enable) {
    const uint32_t address =
        kPlicEnableBase + context * kPlicEnableStride + (source / 32) * static_cast<uint32_t>(sizeof(uint32_t));
    const uint32_t bit = uint32_t{1} << (source % 32);
    const uint32_t word = plic_read32(address);
    plic_write32(address, enable ? (word | bit) : (word & ~bit));
}

// register_handler_for_interrupt writes this slot; reading it first is what lets the teardown put
// back whatever the firmware left there.
inline volatile uint32_t* interrupt_table_slot(uint32_t index) {
    return reinterpret_cast<volatile uint32_t*>(INTERRUPT_TABLE_BASE) + index;
}

// A pending latched before this kernel armed would be delivered the moment the hart enables the
// source and would be indistinguishable from the interrupt under test. Claim and complete until
// nothing is pending sheds them; the PLIC hands out nothing a context has not enabled, so this is
// only meaningful once the enable bits are set.
inline void drain_stale_pendings(uint32_t context) {
    for (uint32_t i = 0; i < kMaxStalePendings; i++) {
        const uint32_t claimed = plic_claim(context);
        if (claimed == 0) {
            return;
        }
        plic_complete(context, claimed);
    }
}

// The ROCC instructions behind the FDS accessors carry no memory clobber, so nothing otherwise
// stops the compiler from sinking a handler's FDS clears past the L1 store that tells the kernel
// the handler ran.
inline void order_fds_before_status() { asm volatile("" ::: "memory"); }

// What arming changed, and everything the teardown needs to put it back. Returned to the kernel
// rather than kept in a static: the handler learns its context from mhartid and its source from
// the claim, so nothing else needs this, and a static would outlive the launch that set it.
struct arming_state {
    uint32_t context = 0;
    // The groups whose PLIC sources were routed to this hart, one bit per group id. More than one
    // is routed when a test has to show which of several configured groups the claim names.
    uint32_t group_mask = 0;
    uint32_t saved_table_word = 0;
};

// Routes every group in group_mask to this hart and points interrupt table slot 11 at handler.
// False means mhartid is outside the PLIC's context range, which would make every context offset a
// guess. The FDS interrupt enable register is deliberately untouched: which groups are armed on the
// FDS side, and when, is what these tests vary.
inline bool arm_external_interrupt(uint32_t group_mask, void (*handler)(), arming_state& state) {
    const uint32_t context = read_mhartid();
    if (context >= kNumPlicContexts) {
        return false;
    }

    state.context = context;
    state.group_mask = group_mask;
    state.saved_table_word = *interrupt_table_slot(MACHINE_EXTERNAL_INTERRUPT_OFFSET);

    register_handler_for_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET, handler);
    // The table is instruction memory: the jump just written is not fetchable until the icache
    // drops what it holds for that address.
    invalidate_l1_icache();

    plic_write32(kPlicThresholdBase + context * kPlicContextStride, 0);
    for (uint32_t mask = group_mask, group = 0; mask != 0; mask >>= 1, group++) {
        if (mask & 1u) {
            const uint32_t source = fds_plic_source(group);
            plic_write32(kPlicPriorityBase + source * static_cast<uint32_t>(sizeof(uint32_t)), kFdsInterruptPriority);
            plic_enable_source(context, source, true);
        }
    }

    drain_stale_pendings(context);

    asm volatile("csrrs zero, mie, %0" ::"r"(kMieMachineExternal));
    asm volatile("csrrs zero, mstatus, %0" ::"r"(kMstatusMie));
    return true;
}

// Everything after the FDS interrupt enable register has been cleared. Split out because that
// first step is the only part of the teardown that differs between the two register maps.
inline void finish_disarm(const arming_state& state) {
    // Drained while the sources are still enabled in this context: the PLIC drops a completion
    // from a context that does not have the source enabled, and a claim left in flight would block
    // that source for every later program on this node.
    drain_stale_pendings(state.context);
    for (uint32_t mask = state.group_mask, group = 0; mask != 0; mask >>= 1, group++) {
        if (mask & 1u) {
            plic_enable_source(state.context, fds_plic_source(group), false);
        }
    }

    asm volatile("csrrc zero, mie, %0" ::"r"(kMieMachineExternal));
    asm volatile("csrrc zero, mstatus, %0" ::"r"(kMstatusMie));

    *interrupt_table_slot(MACHINE_EXTERNAL_INTERRUPT_OFFSET) = state.saved_table_word;
    invalidate_l1_icache();
}

// Called from a handler, before it completes its claim. Nothing happens until the entry count says
// the level is one this kernel cannot lower.
inline void stop_interrupt_storm(uint32_t context, uint32_t source, uint32_t entry) {
    if (entry >= kMaxHandlerEntries) {
        plic_enable_source(context, source, false);
    }
}

// Zero every slot the handler owns, before anything can be delivered into them. The host clears
// the whole block before launch, but these are the only kernels in this suite that read their own
// status block back, and a store from this core is what makes the value definite in the cache this
// core will read it through — whatever an earlier program left cached for the same address. Call
// this before arming. The range is half open: every kernel here owns its slots from the interrupt
// count to the end of its status block, so end_slot is that block's slot count.
inline void clear_handler_slots(fds_kernel::status_ptr status, uint32_t first_slot, uint32_t end_slot) {
    for (uint32_t slot = first_slot; slot < end_slot; slot++) {
        status[slot] = 0;
    }
}

// The interrupt count slot doubles as the handler's flag: the handler bumps it last, so a kernel
// that sees a new value knows the handler ran to the end.
inline bool wait_for_interrupt_count(fds_kernel::status_ptr status, uint32_t expected, uint32_t poll_iterations) {
    for (uint32_t i = 0; i < poll_iterations; i++) {
        if (status[fds_interrupt_status::kSlotInterruptCount] >= expected) {
            return true;
        }
    }
    return false;
}

// True when the count held at expected for the whole window.
inline bool interrupt_count_steady(fds_kernel::status_ptr status, uint32_t expected, uint32_t silence_iterations) {
    for (uint32_t i = 0; i < silence_iterations; i++) {
        if (status[fds_interrupt_status::kSlotInterruptCount] != expected) {
            return false;
        }
    }
    return true;
}

namespace dispatch {

// Clear the input registers currently carrying group_id, which is what lowers the count and with it
// the level. Group status is the live per-lane map and is not gated by the enable register, so it
// names exactly the lanes contributing to the count. A clear sticks under a held sender, so this
// does not have to race the workers.
inline void clear_group_inputs(uint32_t group_id, uint32_t worker_mask) {
    const uint32_t done_lanes = overlay::FdsDispatch::fds_read_group_status(group_id) & worker_mask;
    for (uint32_t mask = done_lanes, neo = 0; mask != 0; mask >>= 1, neo++) {
        if (mask & 1u) {
            overlay::FdsDispatch::fds_clear_neo_status(neo);
        }
    }
}

// Read the count back until it shows the clear has landed. The read back is the point as much as
// the wait is: a completion issued before the level has fallen is re-delivered immediately.
inline void wait_count_cleared(uint32_t group_id) {
    for (uint32_t i = 0; i < kHandlerSettleIterations; i++) {
        if (overlay::FdsDispatch::fds_read_group_count(group_id) == 0) {
            return;
        }
    }
}

inline void disarm_external_interrupt(const arming_state& state) {
    overlay::FdsDispatch::fds_config_interrupt_en(0);
    // Read back so the disable has landed before the drain: a drain that races it would complete
    // claims under a level still standing and shed nothing.
    overlay::FdsDispatch::fds_read_group_count(0);
    finish_disarm(state);
}

}  // namespace dispatch

namespace neo {

// Same read back as the dispatch-side wait, against the register this map exposes.
inline void wait_status_cleared(uint32_t group_id) {
    for (uint32_t i = 0; i < kHandlerSettleIterations; i++) {
        if (overlay::FdsNeo::fds_read_group_status(group_id) == 0) {
            return;
        }
    }
}

inline void disarm_external_interrupt(const arming_state& state) {
    overlay::FdsNeo::fds_config_interrupt_en(0);
    // Same read back as the dispatch-side teardown, for the same reason.
    overlay::FdsNeo::fds_read_group_status(0);
    finish_disarm(state);
}

}  // namespace neo

}  // namespace fds_interrupt
