// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Interrupt delivery on the done threshold: the same handshake the polling tests drive, collected
// by a handler instead of a spin on the count. What this pins beyond delivery itself is the
// identity of what was delivered — the claim must name this group's source and no other, and the
// cause must be a machine external interrupt — and that the level does not come back once the
// handler has cleared the inputs feeding it.
//
// The group is armed before the ready wait, which is only safe because the trigger is an equality:
// the count is zero and the threshold is not, so nothing fires until the dones actually arrive.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"
#include "quasar_fds_interrupt.h"

using fds_interrupt_status::kSlotClaimedSource;
using fds_interrupt_status::kSlotInterruptCount;

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kSlotMcauseLow = 3;
constexpr uint32_t kSlotHartId = 4;
constexpr uint32_t kNumSlots = 5;

// The handler cannot be handed arguments, so it reads the same compile-time values kernel_main
// does. Nothing it needs lives in static storage: its counter and its flag are status slots.
constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kWorkerMask = get_named_compile_time_arg_val("worker_mask");
// The count the interrupt is meant to mark. Every worker in the epoch belongs to this group, so
// the host sets it to the worker count; the two are named separately because the ready wait counts
// workers whatever their group, exactly as in quasar_dispatch_engine_signal.cpp.
constexpr uint32_t kDoneThreshold = get_named_compile_time_arg_val("done_threshold");
constexpr uint32_t kNumReadyWorkers = get_named_compile_time_arg_val("num_workers");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");
constexpr uint32_t kSilenceIterations = get_named_compile_time_arg_val("silence_iterations");

static_assert(kGroupId < kReadyTokenA, "payload group ids must stay below the ready tokens");
static_assert(kDoneThreshold > 0, "a zero threshold would fire the moment the group was armed");

__attribute__((interrupt)) void fds_done_interrupt_handler() {
    fds_kernel::status_ptr status = reinterpret_cast<fds_kernel::status_ptr>(kL1Address);
    const uint32_t context = fds_interrupt::read_mhartid();
    const uint32_t mcause_low = static_cast<uint32_t>(fds_interrupt::read_mcause());
    const uint32_t claimed = fds_interrupt::plic_claim(context);
    const uint32_t entry = status[kSlotInterruptCount];

    status[kSlotHartId] = context;
    status[kSlotMcauseLow] = mcause_low;
    status[kSlotClaimedSource] = claimed;

    // The level has to come down before the claim is completed, and only the inputs feeding the
    // count can bring it down.
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
    // A captured done survives its sender and its program, so shed the previous epoch before an
    // enable bit can turn that stale count into this epoch's interrupt.
    fds_epoch::clear_dispatch_inputs(kWorkerMask);
    overlay::FdsDispatch::fds_read_group_count(kGroupId);
    overlay::FdsDispatch::fds_config_groupid(kGroupId, kWorkerMask, kDoneThreshold);

    fds_interrupt::arming_state arming;
    if (!fds_interrupt::arm_external_interrupt(uint32_t{1} << kGroupId, fds_done_interrupt_handler, arming)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }
    overlay::FdsDispatch::fds_config_interrupt_en(uint32_t{1} << kGroupId);

    if (!fds_kernel::workers_are_ready(status, kL1Address, kNumSlots, kWorkerMask, kNumReadyWorkers, kPollIterations)) {
        fds_interrupt::dispatch::disarm_external_interrupt(arming);
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kGroupId);

    uint32_t result = kComplete;
    if (!fds_interrupt::wait_for_interrupt_count(status, 1, kPollIterations)) {
        result = fds_interrupt_status::kTimeoutInterrupt;
    }

    if (result == kComplete && status[kSlotClaimedSource] != fds_interrupt::fds_plic_source(kGroupId)) {
        result = fds_interrupt_status::kWrongSource;
    }

    if (result == kComplete && status[kSlotMcauseLow] != fds_interrupt::kMachineExternalCauseLow) {
        result = fds_interrupt_status::kWrongCause;
    }

    // The handler cleared the input registers under senders that are still holding their dones. A
    // clear sticks under a held sender, so the count must stay at zero and the level must stay
    // down; a second entry here is the level re-arming itself.
    if (result == kComplete && !fds_interrupt::interrupt_count_steady(status, 1, kSilenceIterations)) {
        result = fds_interrupt_status::kUnexpectedInterrupt;
    }

    fds_interrupt::dispatch::disarm_external_interrupt(arming);
    overlay::FdsDispatch::fds_clear_go();

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
