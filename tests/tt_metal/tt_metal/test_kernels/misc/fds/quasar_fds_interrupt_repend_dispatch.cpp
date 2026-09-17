// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Completing a claim while the level still stands re-delivers it. The handler's first entry
// deliberately skips the clear every other handler in this suite performs and completes with the
// done still on the wire, so the FDS comparison is still true; a second entry must follow. The
// second entry clears, reads back, and completes as usual, and a silence window asserts there is
// no third.
//
// This is the positive counterpart of the silence window in quasar_fds_interrupt_dispatch.cpp:
// that one shows a cleared level staying down, this one shows an uncleared level coming back. Only
// together do they establish that the clear is what quiets the source, rather than the completion.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"
#include "quasar_fds_interrupt.h"

using fds_interrupt_status::kSlotInterruptCount;

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kNumSlots = 2;

// The first delivery never arrived.
constexpr uint32_t kTimeoutFirst = fds_interrupt_status::kTimeoutInterrupt;
// The claim was completed under a standing level and nothing came back.
constexpr uint32_t kTimeoutRedelivery = 0x5A5A0076;
// A third delivery after the second entry cleared the inputs.
constexpr uint32_t kThirdDelivery = fds_interrupt_status::kUnexpectedInterrupt;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kWorkerMask = get_named_compile_time_arg_val("worker_mask");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");
constexpr uint32_t kSilenceIterations = get_named_compile_time_arg_val("silence_iterations");

// One worker, so one done is the whole count.
constexpr uint32_t kDoneThreshold = 1;

static_assert(kGroupId < kReadyTokenA, "payload group ids must stay below the ready tokens");

__attribute__((interrupt)) void fds_repend_interrupt_handler() {
    fds_kernel::status_ptr status = reinterpret_cast<fds_kernel::status_ptr>(kL1Address);
    const uint32_t context = fds_interrupt::read_mhartid();
    const uint32_t claimed = fds_interrupt::plic_claim(context);
    const uint32_t entry = status[kSlotInterruptCount];

    // The first entry leaves the input register alone on purpose, so the count is still at the
    // threshold when the claim is completed below and the source has to re-pend.
    if (entry != 0) {
        fds_interrupt::dispatch::clear_group_inputs(kGroupId, kWorkerMask);
        fds_interrupt::dispatch::wait_count_cleared(kGroupId);
    }
    fds_interrupt::order_fds_before_status();

    fds_interrupt::stop_interrupt_storm(context, claimed, entry);
    fds_interrupt::plic_complete(context, claimed);
    status[kSlotInterruptCount] = entry + 1;
}

void kernel_main() {
    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(kL1Address, kNumSlots);
    fds_interrupt::clear_handler_slots(status, kSlotInterruptCount, kNumSlots);
    // The first entry deliberately leaves its input standing, so it must be caused by this epoch's
    // done rather than a capture inherited from an earlier launch.
    fds_epoch::clear_dispatch_inputs(kWorkerMask);
    overlay::FdsDispatch::fds_read_group_count(kGroupId);
    overlay::FdsDispatch::fds_config_groupid(kGroupId, kWorkerMask, kDoneThreshold);

    fds_interrupt::arming_state arming;
    if (!fds_interrupt::arm_external_interrupt(uint32_t{1} << kGroupId, fds_repend_interrupt_handler, arming)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }
    overlay::FdsDispatch::fds_config_interrupt_en(uint32_t{1} << kGroupId);

    if (!fds_kernel::workers_are_ready(status, kL1Address, kNumSlots, kWorkerMask, kNumWorkers, kPollIterations)) {
        fds_interrupt::dispatch::disarm_external_interrupt(arming);
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kGroupId);

    uint32_t result = kComplete;
    if (!fds_interrupt::wait_for_interrupt_count(status, 1, kPollIterations)) {
        result = kTimeoutFirst;
    }

    // Both entries can land before this kernel runs again, so the wait is for the count reaching
    // two rather than for a transition through one.
    if (result == kComplete && !fds_interrupt::wait_for_interrupt_count(status, 2, kPollIterations)) {
        result = kTimeoutRedelivery;
    }

    if (result == kComplete && !fds_interrupt::interrupt_count_steady(status, 2, kSilenceIterations)) {
        result = kThirdDelivery;
    }

    fds_interrupt::dispatch::disarm_external_interrupt(arming);
    overlay::FdsDispatch::fds_clear_go();

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
