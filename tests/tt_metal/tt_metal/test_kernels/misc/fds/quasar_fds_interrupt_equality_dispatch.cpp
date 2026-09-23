// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The trigger is an equality, not a meets-or-exceeds. The threshold is set one below the number of
// workers, so the count overshoots it and stops there, and the interrupt enable is withheld until
// it has: arming against a count of N with a threshold of N-1 must stay silent. A model that fires
// on count >= threshold fires here.
//
// The count is then brought down onto the threshold by clearing one done lane — the count is
// derived from the input registers, so dropping one lane subtracts one from it — and the same
// arming that was silent a moment ago must now deliver. The two halves are what make this an
// assertion about the comparison rather than about arming order: nothing between them changes but
// the count, and it changes downwards.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"
#include "quasar_fds_interrupt.h"

using fds_interrupt_status::kSlotInterruptCount;

// Mirrored by test_quasar_fds.cpp. The lane map the group was holding when the enable bit went on,
// so a failure names the count the comparison was made against.
constexpr uint32_t kSlotStatusAtArm = 2;
constexpr uint32_t kNumSlots = 3;

// The dones never all showed on the wire, so the overshoot under test never existed.
constexpr uint32_t kTimeoutStatus = 0x5A5A0075;
// Fired while the count sat above the threshold: a meets-or-exceeds trigger.
constexpr uint32_t kFiredAtOvershoot = fds_interrupt_status::kUnexpectedInterrupt;
// The count was brought down onto the threshold and nothing fired.
constexpr uint32_t kTimeoutAtEquality = fds_interrupt_status::kTimeoutInterrupt;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kWorkerMask = get_named_compile_time_arg_val("worker_mask");
constexpr uint32_t kNumReadyWorkers = get_named_compile_time_arg_val("num_workers");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");
constexpr uint32_t kSilenceIterations = get_named_compile_time_arg_val("silence_iterations");

// One below the number of workers, so the count can reach the threshold only by coming back down.
constexpr uint32_t kDoneThreshold = kNumReadyWorkers - 1;

static_assert(kGroupId < kReadyTokenA, "payload group ids must stay below the ready tokens");
static_assert(kNumReadyWorkers >= 2, "the overshoot needs a threshold of at least one below N");

__attribute__((interrupt)) void fds_equality_interrupt_handler() {
    fds_kernel::status_ptr status = reinterpret_cast<fds_kernel::status_ptr>(kL1Address);
    const uint32_t context = fds_interrupt::read_mhartid();
    const uint32_t claimed = fds_interrupt::plic_claim(context);
    const uint32_t entry = status[kSlotInterruptCount];

    fds_interrupt::dispatch::clear_group_inputs(kGroupId, kWorkerMask);
    fds_interrupt::dispatch::wait_count_cleared(kGroupId);
    fds_interrupt::order_fds_before_status();

    fds_interrupt::stop_interrupt_storm(context, claimed, entry);
    fds_interrupt::plic_complete(context, claimed);
    status[kSlotInterruptCount] = entry + 1;
}

void kernel_main() {
    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(kL1Address, kNumSlots);
    fds_interrupt::clear_handler_slots(status, kSlotInterruptCount, kNumSlots);
    overlay::FdsDispatch::fds_config_groupid(kGroupId, kWorkerMask, kDoneThreshold);

    // The PLIC side is armed up front; the FDS interrupt enable is what this test withholds.
    fds_interrupt::arming_state arming;
    if (!fds_interrupt::arm_external_interrupt(uint32_t{1} << kGroupId, fds_equality_interrupt_handler, arming)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }

    if (!fds_kernel::workers_are_ready(status, kL1Address, kNumSlots, kWorkerMask, kNumReadyWorkers, kPollIterations)) {
        fds_interrupt::dispatch::disarm_external_interrupt(arming);
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kGroupId);

    // Waited out on status rather than on the count, because status is ungated by the enable mask
    // and so reports the lanes whether or not the group was ever configured to count them. Every
    // lane present is the overshoot the arming below has to be silent against.
    uint32_t result = kComplete;
    uint32_t status_at_arm = 0;
    bool all_dones_visible = false;
    for (uint32_t i = 0; i < kPollIterations && !all_dones_visible; i++) {
        status_at_arm = overlay::FdsDispatch::fds_read_group_status(kGroupId) & kWorkerMask;
        all_dones_visible = static_cast<uint32_t>(__builtin_popcount(status_at_arm)) >= kNumReadyWorkers;
    }
    status[kSlotStatusAtArm] = status_at_arm;
    if (!all_dones_visible) {
        result = kTimeoutStatus;
    }

    if (result == kComplete) {
        overlay::FdsDispatch::fds_config_interrupt_en(uint32_t{1} << kGroupId);
        if (!fds_interrupt::interrupt_count_steady(status, 0, kSilenceIterations)) {
            result = kFiredAtOvershoot;
        }
    }

    if (result == kComplete) {
        // One lane out of the count, and the arming that was silent must now deliver.
        const uint32_t cleared_lane = static_cast<uint32_t>(__builtin_ctz(status_at_arm));
        overlay::FdsDispatch::fds_clear_neo_status(cleared_lane);
        if (!fds_interrupt::wait_for_interrupt_count(status, 1, kPollIterations)) {
            result = kTimeoutAtEquality;
        }
    }

    fds_interrupt::dispatch::disarm_external_interrupt(arming);
    overlay::FdsDispatch::fds_clear_go();

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
