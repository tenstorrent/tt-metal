// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A program that stops inside a handler, on an assert for one, leaves its PLIC claim open, and the
// PLIC holds that source back until the claim is completed. The next program to register the source
// would then never see it. Registering a handler therefore completes the source, and this kernel
// checks that it does.
//
// The open claim is made by hand, with the machine-external interrupt disabled: the source is
// enabled, raised, claimed, and never completed. Before anything is registered, the kernel raises
// the source again and checks that no claim can be made. If one can, this PLIC does not hold a
// source back behind an open claim, there is nothing for a registration to release, and the kernel
// reports kClaimDoesNotBlock for the host test to skip on.
//
// Then the source is registered and raised once more, and it has to reach the handler.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_external_interrupt.h"

using fds_interrupt_status::kSlotClaimedSource;
using fds_interrupt_status::kSlotInterruptCount;

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kNumSlots = 3;

// The source was raised and no claim could be made, so there was no claim to leave open.
constexpr uint32_t kTimeoutClaim = 0x5A5A00A0;
// The source was claimed again while its first claim was still open.
constexpr uint32_t kClaimDoesNotBlock = 0x5A5A00A1;
// The source was registered and still never reached its handler.
constexpr uint32_t kStaleClaimNotReleased = 0x5A5A00A2;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");
constexpr uint32_t kSilenceIterations = get_named_compile_time_arg_val("silence_iterations");

static_assert(kGroupId != 0 && kGroupId < kReadyTokenA, "the group id must be a payload group");

constexpr uint32_t kGroup = external_interrupt::group_bit(kGroupId);
constexpr uint32_t kSource = external_interrupt::source_for_group(kGroupId);

void source_handler(uint32_t plic_source) {
    external_interrupt::lower_levels_and_record(
        reinterpret_cast<fds_kernel::status_ptr>(kL1Address), kSlotInterruptCount, kSlotClaimedSource, plic_source);
}

void kernel_main() {
    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(kL1Address, kNumSlots);
    fds_interrupt::clear_handler_slots(status, kSlotInterruptCount, kNumSlots);
    external_interrupt::configure_self_raising_groups(kGroup);

    // Installing does not arm: mie.MEIE stays clear until arm_machine_external_interrupts, so every
    // claim below is the kernel's own.
    fds_interrupt::arming_state installed;
    if (!external_interrupt::install_dispatcher(kGroup, installed)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }

    overlay::quasar::plic_set_priority(kSource, external_interrupt::kPriority);
    overlay::quasar::plic_enable_source(kSource);
    fds_interrupt::drain_stale_pendings(external_interrupt::kPlicContext);

    // Claim the source and leave the claim open.
    uint32_t result = kComplete;
    external_interrupt::raise_levels(kGroup);
    const uint32_t claimed = external_interrupt::poll_claim(kPollIterations);
    external_interrupt::lower_levels();
    if (claimed != kSource) {
        result = kTimeoutClaim;
        if (claimed != overlay::quasar::plic_no_source) {
            overlay::quasar::plic_complete(claimed);
        }
    }

    // The open claim has to hold the source back, or the rest of this kernel shows nothing.
    if (result == kComplete) {
        external_interrupt::raise_levels(kGroup);
        const uint32_t claimed_again = external_interrupt::poll_claim(kSilenceIterations);
        external_interrupt::lower_levels();
        if (claimed_again != overlay::quasar::plic_no_source) {
            overlay::quasar::plic_complete(claimed_again);
            result = kClaimDoesNotBlock;
        }
    }

    // Registering has to complete the open claim: the level raised next is delivered only if it did.
    // The drain is the one the firmware runs between registering and arming, so that the count
    // below is this kernel's own level and nothing the completion let through.
    if (result == kComplete) {
        register_machine_external_interrupt_handler(kSource, external_interrupt::kPriority, source_handler);
        fds_interrupt::drain_stale_pendings(external_interrupt::kPlicContext);
        arm_machine_external_interrupts();
        external_interrupt::raise_levels(kGroup);
        if (!external_interrupt::wait_for_slot(status, kSlotInterruptCount, 1, kPollIterations)) {
            result = kStaleClaimNotReleased;
        } else if (status[kSlotClaimedSource] != kSource) {
            result = fds_interrupt_status::kWrongSource;
        }
    }
    if (result == kComplete && !external_interrupt::slot_steady(status, kSlotInterruptCount, 1, kSilenceIterations)) {
        result = fds_interrupt_status::kUnexpectedInterrupt;
    }

    // The claim is still open, and would hold the source back in every later program on this node.
    if (result == kStaleClaimNotReleased) {
        overlay::quasar::plic_complete(kSource);
    }

    external_interrupt::remove_dispatcher(installed);

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
