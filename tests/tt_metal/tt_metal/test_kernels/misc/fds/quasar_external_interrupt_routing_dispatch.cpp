// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two PLIC sources share the machine-external interrupt, each with its own registered handler. The
// dispatcher has to hand each claim to the handler registered for that source and to no other, and
// complete the claim afterwards. The steps, in order:
//
// - Registering the second source leaves the first enabled.
// - Each source raised alone reaches only its own handler, which is handed that source.
// - A source raised a second time is delivered again. A claim the dispatcher had left open would
//   block it for good.
// - Both sources raised by one write reach both handlers, once each.
// - An unregistered source is disabled, the other is not, and raising it calls nothing.
// - disable_machine_interrupt holds a pending source back without clearing mstatus.MIE, and
//   enable_machine_interrupt lets it through.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_external_interrupt.h"

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kSlotFirstCount = 1;
constexpr uint32_t kSlotFirstSource = 2;
constexpr uint32_t kSlotSecondCount = 3;
constexpr uint32_t kSlotSecondSource = 4;
constexpr uint32_t kNumSlots = 5;

// Registering the second source cost the first its enable bit, or one was never enabled.
constexpr uint32_t kEnableLost = 0x5A5A0080;
// Arming left mie.MEIE or mstatus.MIE clear.
constexpr uint32_t kNotArmed = 0x5A5A0081;
// A source raised alone never reached its handler.
constexpr uint32_t kTimeoutFirst = 0x5A5A0082;
constexpr uint32_t kTimeoutSecond = 0x5A5A0083;
// A handler ran for the other source, or was handed a source other than its own.
constexpr uint32_t kWrongHandler = 0x5A5A0084;
// The first source was not delivered a second time, so its first claim was never completed.
constexpr uint32_t kNotCompleted = 0x5A5A0085;
// Both sources were raised together and one of them never reached its handler.
constexpr uint32_t kTimeoutBoth = 0x5A5A0086;
// Unregistering left the source enabled, or took the other source's enable bit with it.
constexpr uint32_t kUnregisterEnable = 0x5A5A0087;
// The unregistered source still reached a handler.
constexpr uint32_t kDeliveredAfterUnregister = 0x5A5A0088;
// disable_machine_interrupt left mie.MEIE set, or cleared mstatus.MIE.
constexpr uint32_t kDisableCsr = 0x5A5A0089;
// A source was delivered while the machine-external interrupt was disabled.
constexpr uint32_t kDeliveredWhileDisabled = 0x5A5A008A;
// The source held back while disabled was not delivered once the interrupt was enabled again.
constexpr uint32_t kNotDeliveredAfterEnable = 0x5A5A008B;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kFirstGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kSecondGroupId = get_named_compile_time_arg_val("second_group_id");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");
constexpr uint32_t kSilenceIterations = get_named_compile_time_arg_val("silence_iterations");

static_assert(kFirstGroupId != 0 && kFirstGroupId < kReadyTokenA, "group ids must be payload groups");
static_assert(kSecondGroupId != 0 && kSecondGroupId < kReadyTokenA, "group ids must be payload groups");
static_assert(kFirstGroupId != kSecondGroupId, "the two groups must be separate PLIC sources");

constexpr uint32_t kFirstGroup = external_interrupt::group_bit(kFirstGroupId);
constexpr uint32_t kSecondGroup = external_interrupt::group_bit(kSecondGroupId);
constexpr uint32_t kFirstSource = external_interrupt::source_for_group(kFirstGroupId);
constexpr uint32_t kSecondSource = external_interrupt::source_for_group(kSecondGroupId);

void first_source_handler(uint32_t plic_source) {
    external_interrupt::lower_levels_and_record(
        reinterpret_cast<fds_kernel::status_ptr>(kL1Address), kSlotFirstCount, kSlotFirstSource, plic_source);
}

void second_source_handler(uint32_t plic_source) {
    external_interrupt::lower_levels_and_record(
        reinterpret_cast<fds_kernel::status_ptr>(kL1Address), kSlotSecondCount, kSlotSecondSource, plic_source);
}

void kernel_main() {
    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(kL1Address, kNumSlots);
    fds_interrupt::clear_handler_slots(status, kSlotFirstCount, kNumSlots);
    external_interrupt::configure_self_raising_groups(kFirstGroup | kSecondGroup);

    fds_interrupt::arming_state installed;
    if (!external_interrupt::install_dispatcher(kFirstGroup | kSecondGroup, installed)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }

    register_machine_external_interrupt_handler(kFirstSource, external_interrupt::kPriority, first_source_handler);
    register_machine_external_interrupt_handler(kSecondSource, external_interrupt::kPriority, second_source_handler);

    uint32_t result = kComplete;
    if (!external_interrupt::source_enabled(kFirstSource) || !external_interrupt::source_enabled(kSecondSource)) {
        result = kEnableLost;
    }

    // A pending latched by an earlier program would be delivered as this kernel's first interrupt.
    fds_interrupt::drain_stale_pendings(external_interrupt::kPlicContext);
    arm_machine_external_interrupts();
    if (result == kComplete && ((external_interrupt::read_mie() & external_interrupt::kMieMachineExternal) == 0 ||
                                (external_interrupt::read_mstatus() & MSTATUS_MIE) == 0)) {
        result = kNotArmed;
    }

    // Each source alone.
    if (result == kComplete) {
        external_interrupt::raise_levels(kFirstGroup);
        if (!external_interrupt::wait_for_slot(status, kSlotFirstCount, 1, kPollIterations)) {
            result = kTimeoutFirst;
        } else if (status[kSlotFirstSource] != kFirstSource || status[kSlotSecondCount] != 0) {
            result = kWrongHandler;
        }
    }
    if (result == kComplete) {
        external_interrupt::raise_levels(kSecondGroup);
        if (!external_interrupt::wait_for_slot(status, kSlotSecondCount, 1, kPollIterations)) {
            result = kTimeoutSecond;
        } else if (status[kSlotSecondSource] != kSecondSource || status[kSlotFirstCount] != 1) {
            result = kWrongHandler;
        }
    }

    // The first source again. It is delivered only if the dispatcher completed its first claim.
    if (result == kComplete) {
        external_interrupt::raise_levels(kFirstGroup);
        if (!external_interrupt::wait_for_slot(status, kSlotFirstCount, 2, kPollIterations)) {
            result = kNotCompleted;
        }
    }

    // Both by one write. Whichever handler runs first lowers both levels, and the other source is
    // already latched pending, so the dispatcher still has a second claim to hand out.
    if (result == kComplete) {
        external_interrupt::raise_levels(kFirstGroup | kSecondGroup);
        if (!external_interrupt::wait_for_slot(status, kSlotFirstCount, 3, kPollIterations) ||
            !external_interrupt::wait_for_slot(status, kSlotSecondCount, 2, kPollIterations)) {
            result = kTimeoutBoth;
        }
    }
    if (result == kComplete && (!external_interrupt::slot_steady(status, kSlotFirstCount, 3, kSilenceIterations) ||
                                !external_interrupt::slot_steady(status, kSlotSecondCount, 2, kSilenceIterations))) {
        result = fds_interrupt_status::kUnexpectedInterrupt;
    }

    // Unregistering stops delivery at the PLIC. The level raised here stays pending on the disabled
    // source, which the teardown sheds.
    if (result == kComplete) {
        unregister_machine_external_interrupt_handler(kSecondSource);
        if (external_interrupt::source_enabled(kSecondSource) || !external_interrupt::source_enabled(kFirstSource)) {
            result = kUnregisterEnable;
        }
    }
    if (result == kComplete) {
        external_interrupt::raise_levels(kSecondGroup);
        if (!external_interrupt::slot_steady(status, kSlotSecondCount, 2, kSilenceIterations)) {
            result = kDeliveredAfterUnregister;
        }
        external_interrupt::lower_levels();
    }

    // The mie helpers. The source raised while the interrupt is disabled stays pending at the PLIC,
    // and enabling the interrupt is what delivers it.
    if (result == kComplete) {
        disable_machine_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET);
        if ((external_interrupt::read_mie() & external_interrupt::kMieMachineExternal) != 0 ||
            (external_interrupt::read_mstatus() & MSTATUS_MIE) == 0) {
            result = kDisableCsr;
        }
    }
    if (result == kComplete) {
        external_interrupt::raise_levels(kFirstGroup);
        if (!external_interrupt::slot_steady(status, kSlotFirstCount, 3, kSilenceIterations)) {
            result = kDeliveredWhileDisabled;
        }
    }
    if (result == kComplete) {
        enable_machine_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET);
        if (!external_interrupt::wait_for_slot(status, kSlotFirstCount, 4, kPollIterations)) {
            result = kNotDeliveredAfterEnable;
        }
    }

    external_interrupt::remove_dispatcher(installed);

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
