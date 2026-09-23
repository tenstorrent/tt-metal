// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The interrupt enable register is per group, and the source the PLIC reports is the group that
// fired. Two groups are configured identically — full lane mask, threshold equal to the number of
// workers — and only one of them is ever signalled. The quiet group is armed and the signalled one
// is not, so during the silence window the signalled group sits exactly at its threshold with no
// enable bit while the armed group sits at a count of zero against a threshold of N. Neither may
// deliver, and a per-group enable is the only reason.
//
// The signalled group is then armed too, and the claim must name its source rather than the quiet
// group's. Both sources are routed to this hart at the PLIC, so naming the wrong one is a result
// this test can actually observe.
//
// Distinct from quasar_fds_interrupt_equality_dispatch.cpp: that one varies the count against a
// fixed arming, this one varies the arming against a fixed count.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"
#include "quasar_fds_interrupt.h"

using fds_interrupt_status::kSlotClaimedSource;
using fds_interrupt_status::kSlotInterruptCount;

// Mirrored by test_quasar_fds.cpp.
constexpr uint32_t kNumSlots = 3;

// The dones never all showed on the wire, so the signalled group never reached its threshold.
constexpr uint32_t kTimeoutStatus = 0x5A5A0079;
// A delivery while the only armed group was nowhere near its threshold.
constexpr uint32_t kFiredUnarmed = fds_interrupt_status::kUnexpectedInterrupt;

constexpr uint32_t kL1Address = get_named_compile_time_arg_val("l1_address");
constexpr uint32_t kGroupId = get_named_compile_time_arg_val("group_id");
constexpr uint32_t kQuietGroupId = get_named_compile_time_arg_val("quiet_group_id");
constexpr uint32_t kWorkerMask = get_named_compile_time_arg_val("worker_mask");
constexpr uint32_t kNumReadyWorkers = get_named_compile_time_arg_val("num_workers");
constexpr uint32_t kPollIterations = get_named_compile_time_arg_val("poll_iterations");
constexpr uint32_t kSilenceIterations = get_named_compile_time_arg_val("silence_iterations");

static_assert(kGroupId < kReadyTokenA, "payload group ids must stay below the ready tokens");
static_assert(kQuietGroupId < kReadyTokenA, "payload group ids must stay below the ready tokens");
static_assert(kGroupId != kQuietGroupId, "the two groups must be distinguishable in the claim");
static_assert(kNumReadyWorkers > 0, "a zero threshold would fire both groups before any done arrived");

__attribute__((interrupt)) void fds_group_mask_interrupt_handler() {
    fds_kernel::status_ptr status = reinterpret_cast<fds_kernel::status_ptr>(kL1Address);
    const uint32_t context = fds_interrupt::read_mhartid();
    const uint32_t claimed = fds_interrupt::plic_claim(context);
    const uint32_t entry = status[kSlotInterruptCount];

    status[kSlotClaimedSource] = claimed;

    // Only the signalled group has lanes to clear; the quiet group shares the lanes but no worker
    // ever drove its id onto one, so its count is zero throughout and clearing does nothing to it.
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
    // Both groups participate in a silence assertion, so neither may begin with a captured value
    // from a previous launch.
    fds_epoch::clear_dispatch_inputs(kWorkerMask);
    overlay::FdsDispatch::fds_read_group_count(kGroupId);
    overlay::FdsDispatch::fds_config_groupid(kGroupId, kWorkerMask, kNumReadyWorkers);
    overlay::FdsDispatch::fds_config_groupid(kQuietGroupId, kWorkerMask, kNumReadyWorkers);

    // Both sources routed to this hart, so the claim can name either one.
    fds_interrupt::arming_state arming;
    const uint32_t routed_groups = (uint32_t{1} << kGroupId) | (uint32_t{1} << kQuietGroupId);
    if (!fds_interrupt::arm_external_interrupt(routed_groups, fds_group_mask_interrupt_handler, arming)) {
        fds_kernel::finish(status, kL1Address, kNumSlots, fds_interrupt_status::kBadHartContext);
        return;
    }
    // On the FDS side, only the group nobody will signal.
    overlay::FdsDispatch::fds_config_interrupt_en(uint32_t{1} << kQuietGroupId);

    if (!fds_kernel::workers_are_ready(status, kL1Address, kNumSlots, kWorkerMask, kNumReadyWorkers, kPollIterations)) {
        fds_interrupt::dispatch::disarm_external_interrupt(arming);
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kGroupId);

    // Status rather than count, so the wait does not depend on the enable register this test is
    // manipulating.
    uint32_t result = kComplete;
    bool all_dones_visible = false;
    for (uint32_t i = 0; i < kPollIterations && !all_dones_visible; i++) {
        const uint32_t done_lanes = overlay::FdsDispatch::fds_read_group_status(kGroupId) & kWorkerMask;
        all_dones_visible = static_cast<uint32_t>(__builtin_popcount(done_lanes)) >= kNumReadyWorkers;
    }
    if (!all_dones_visible) {
        result = kTimeoutStatus;
    }

    if (result == kComplete && !fds_interrupt::interrupt_count_steady(status, 0, kSilenceIterations)) {
        result = kFiredUnarmed;
    }

    if (result == kComplete) {
        overlay::FdsDispatch::fds_config_interrupt_en(routed_groups);
        if (!fds_interrupt::wait_for_interrupt_count(status, 1, kPollIterations)) {
            result = fds_interrupt_status::kTimeoutInterrupt;
        }
    }

    if (result == kComplete && status[kSlotClaimedSource] != fds_interrupt::fds_plic_source(kGroupId)) {
        result = fds_interrupt_status::kWrongSource;
    }

    fds_interrupt::dispatch::disarm_external_interrupt(arming);
    overlay::FdsDispatch::fds_clear_go();

    fds_kernel::finish(status, kL1Address, kNumSlots, result);
}
