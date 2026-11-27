// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A PLIC source that is enabled but has no registered handler. Nothing can lower a level the
// dispatcher knows nothing about, so it has to stop delivery itself: complete the claim, disable the
// source, and report. A dispatcher that only completed would be re-entered for ever, and one that
// left the claim open would block the source for every later program.
//
// One group is registered and one is only enabled at the PLIC. The unregistered one is raised and
// never lowered by a handler. The kernel survives that only if the dispatcher disabled the source;
// the registered source must keep its enable bit and still be delivered afterwards.
//
// That the claim was completed is read from the PLIC: once the source is enabled and raised again, a
// claim made by hand has to return it. The PLIC holds a source back behind a claim left open, and
// the hand claim would find nothing.
//
// The dispatcher also asserts, which hangs the core where asserts are compiled in. The kernel then
// reports kAssertsEnabled without raising anything, and the host test skips.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_external_interrupt.h"

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kSlotRegisteredCount = 1;
constexpr uint32_t kSlotRegisteredSource = 2;
constexpr uint32_t kNumSlots = 3;

// Asserts are compiled in, so the dispatcher would hang on the unregistered source.
constexpr uint32_t kAssertsEnabled = 0x5A5A0090;
// The unregistered source was raised and kept its enable bit.
constexpr uint32_t kNotDisabled = 0x5A5A0091;
// The registered handler ran for the unregistered source.
constexpr uint32_t kWrongHandler = 0x5A5A0092;
// Disabling the unregistered source took the registered source's enable bit with it.
constexpr uint32_t kDisabledRegistered = 0x5A5A0093;
// The registered source was not delivered after the unregistered one had been handled.
constexpr uint32_t kTimeoutRegistered = 0x5A5A0094;
// The unregistered source could not be claimed again, so its claim was never completed.
constexpr uint32_t kClaimLeftOpen = 0x5A5A0095;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kRegisteredGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kUnregisteredGroupId = get_named_compile_time_arg_val("second_group_id");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");

static_assert(kRegisteredGroupId != 0 && kRegisteredGroupId < kReadyTokenA, "group ids must be payload groups");
static_assert(kUnregisteredGroupId != 0 && kUnregisteredGroupId < kReadyTokenA, "group ids must be payload groups");
static_assert(kRegisteredGroupId != kUnregisteredGroupId, "the two groups must be separate PLIC sources");

constexpr uint32_t kRegisteredGroup = external_interrupt::group_bit(kRegisteredGroupId);
constexpr uint32_t kUnregisteredGroup = external_interrupt::group_bit(kUnregisteredGroupId);
constexpr uint32_t kRegisteredSource = external_interrupt::source_for_group(kRegisteredGroupId);
constexpr uint32_t kUnregisteredSource = external_interrupt::source_for_group(kUnregisteredGroupId);

constexpr bool kAssertsCompiledIn = ASSERT_ENABLED != 0;

void registered_source_handler(uint32_t plic_source) {
    external_interrupt::lower_levels_and_record(
        reinterpret_cast<fds_kernel::status_ptr>(kL1Address), kSlotRegisteredCount, kSlotRegisteredSource, plic_source);
}

void kernel_main() {
    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(kL1Address, kNumSlots);
    fds_interrupt::clear_handler_slots(status, kSlotRegisteredCount, kNumSlots);
    if (kAssertsCompiledIn) {
        fds_kernel::finish(status, kL1Address, kNumSlots, kAssertsEnabled);
        return;
    }
    external_interrupt::configure_self_raising_groups(kRegisteredGroup | kUnregisteredGroup);

    fds_interrupt::arming_state installed;
    if (!external_interrupt::install_dispatcher(kRegisteredGroup | kUnregisteredGroup, installed)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }

    register_machine_external_interrupt_handler(
        kRegisteredSource, external_interrupt::kPriority, registered_source_handler);
    // Enabled the way a registration would enable it, but with no handler behind it.
    overlay::quasar::plic_set_priority(kUnregisteredSource, external_interrupt::kPriority);
    overlay::quasar::plic_enable_source(kUnregisteredSource);

    fds_interrupt::drain_stale_pendings(external_interrupt::kPlicContext);
    arm_machine_external_interrupts();

    // Nothing lowers this level, so the kernel gets past the next line only once the dispatcher has
    // disabled the source.
    external_interrupt::raise_levels(kUnregisteredGroup);

    uint32_t result = kNotDisabled;
    for (uint32_t i = 0; i < kPollIterations && result != kComplete; i++) {
        if (!external_interrupt::source_enabled(kUnregisteredSource)) {
            result = kComplete;
        }
    }
    if (result == kComplete && status[kSlotRegisteredCount] != 0) {
        result = kWrongHandler;
    }
    if (result == kComplete && !external_interrupt::source_enabled(kRegisteredSource)) {
        result = kDisabledRegistered;
    }

    // The line still delivers. This write also lowers the unregistered group's level.
    if (result == kComplete) {
        external_interrupt::raise_levels(kRegisteredGroup);
        if (!external_interrupt::wait_for_slot(status, kSlotRegisteredCount, 1, kPollIterations)) {
            result = kTimeoutRegistered;
        } else if (status[kSlotRegisteredSource] != kRegisteredSource) {
            result = kWrongHandler;
        }
    }

    // The source is enabled and raised again, and claimed by hand, with the interrupt disabled so the
    // dispatcher cannot take the claim first. The level is lowered before the completion, so this
    // claim leaves nothing pending behind it.
    if (result == kComplete) {
        disable_machine_interrupt(MACHINE_EXTERNAL_INTERRUPT_OFFSET);
        overlay::quasar::plic_enable_source(kUnregisteredSource);
        external_interrupt::raise_levels(kUnregisteredGroup);
        const uint32_t claimed = external_interrupt::poll_claim(kPollIterations);
        external_interrupt::lower_levels();
        if (claimed != kUnregisteredSource) {
            result = kClaimLeftOpen;
        }
        if (claimed != overlay::quasar::plic_no_source) {
            overlay::quasar::plic_complete(claimed);
        }
    }

    external_interrupt::remove_dispatcher(installed);

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
