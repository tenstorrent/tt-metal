// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the derivation-semantics test. Status, count and the interrupt are all
// recomputed from the input registers every cycle; nothing accumulates. Three consequences are
// asserted here, against workers that raise their dones and hold them (the standard worker kernel,
// quasar_fds_worker_signal.cpp):
//  1. The enable mask filters counting only, never status: with the mask empty, status shows every
//     done and the count reads zero.
//  2. The count is live and falls as well as rises: clearing the input registers returns it to
//     zero while the workers still hold their dones, and it stays there because a clear sticks.
//  3. Group 0 is nothing but the idle decode: it complements the busy map while the dones are
//     held, and reads all-ones over the source width once every input register is clear.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

// Extra status slots, mirrored by the host in test_quasar_dispatch_engines.cpp.
// The count read back while the enable mask was empty; anything but zero is the failure.
constexpr uint32_t kSlotCountUnderEmptyEnable = 1;
// The group status observed after the input registers were cleared under held senders.
constexpr uint32_t kSlotStatusAfterClear = 2;
// The group 0 status, recorded when either group 0 check fails.
constexpr uint32_t kSlotIdleStatus = 3;
constexpr uint32_t kNumSlots = 4;
// The dones never all showed in status.
constexpr uint32_t kTimeoutStatus = 0x5A5A0005;
// The count was nonzero although the group's enable mask was empty.
constexpr uint32_t kCountedWithoutEnable = 0x5A5A0006;
// The count never reached the worker total once the enable mask was set.
constexpr uint32_t kTimeoutCount = 0x5A5A0007;
// Status or count never fell to zero after the input registers were cleared.
constexpr uint32_t kTimeoutClear = 0x5A5A0008;
// Status or count came back while the input registers stayed cleared under held senders.
constexpr uint32_t kRelatchedAfterClear = 0x5A5A0009;
// Group 0's status was not the all-idle map.
constexpr uint32_t kBadIdleMap = 0x5A5A000A;
// Group 0's status did not complement the busy map while the dones were held.
constexpr uint32_t kBadBusyIdleMap = 0x5A5A000B;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t num_workers = get_named_compile_time_arg_val("num_workers");
    constexpr uint32_t silence_iterations = get_named_compile_time_arg_val("silence_iterations");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    // The enable mask is deliberately left empty: status must show the dones anyway, and the count
    // must not.
    overlay::FdsDispatch::fds_config_groupid(group_id, 0, 0);

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, num_workers, poll_iterations)) {
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

    uint32_t result = kComplete;

    // Status is not gated by the enable mask, so every done must show without any configuration.
    bool all_dones_visible = false;
    for (uint32_t i = 0; i < poll_iterations && !all_dones_visible; i++) {
        const uint32_t done_lanes = overlay::FdsDispatch::fds_read_group_status(group_id) & worker_mask;
        all_dones_visible = static_cast<uint32_t>(__builtin_popcount(done_lanes)) >= num_workers;
    }
    if (!all_dones_visible) {
        result = kTimeoutStatus;
    }

    if (result == kComplete) {
        const uint32_t count = overlay::FdsDispatch::fds_read_group_count(group_id);
        status[kSlotCountUnderEmptyEnable] = count;
        if (count != 0) {
            result = kCountedWithoutEnable;
        }
    }

    if (result == kComplete) {
        overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, num_workers);
        uint32_t count = 0;
        if (!fds_kernel::wait_group_count(group_id, num_workers, poll_iterations, count)) {
            result = kTimeoutCount;
        }
    }

    if (result == kComplete) {
        // With the dones still held, group 0 must read as the exact complement of the busy lanes:
        // the idle map is a live decode, not a constant that happens to be all-ones when idle.
        const uint32_t busy_lanes = overlay::FdsDispatch::fds_read_group_status(group_id);
        const uint32_t idle_lanes = overlay::FdsDispatch::fds_read_group_status(0);
        if ((busy_lanes ^ idle_lanes) != 0xFFFFFFFF) {
            status[kSlotIdleStatus] = idle_lanes;
            result = kBadBusyIdleMap;
        }
    }

    if (result == kComplete) {
        // Clear every input register while the workers still hold their dones. The count must fall
        // to zero and stay there: it is derived, and a clear sticks under a held sender.
        fds_epoch::clear_dispatch_inputs(worker_mask);
        bool fell = false;
        for (uint32_t i = 0; i < poll_iterations && !fell; i++) {
            fell = (overlay::FdsDispatch::fds_read_group_status(group_id) & worker_mask) == 0 &&
                   overlay::FdsDispatch::fds_read_group_count(group_id) == 0;
        }
        if (!fell) {
            result = kTimeoutClear;
        }
        for (uint32_t i = 0; i < silence_iterations && result == kComplete; i++) {
            const uint32_t group_status = overlay::FdsDispatch::fds_read_group_status(group_id) & worker_mask;
            if (group_status != 0 || overlay::FdsDispatch::fds_read_group_count(group_id) != 0) {
                status[kSlotStatusAfterClear] = group_status;
                result = kRelatchedAfterClear;
            }
        }
    }

    if (result == kComplete) {
        // With every input register clear, group 0's status is the map of quiet lanes: all of
        // them, tied-off lanes included.
        const uint32_t idle_status = overlay::FdsDispatch::fds_read_group_status(0);
        status[kSlotIdleStatus] = idle_status;
        if (idle_status != 0xFFFFFFFF) {
            result = kBadIdleMap;
        }
    }

    overlay::FdsDispatch::fds_clear_go();
    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
