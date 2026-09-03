// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

// Mirrored by test_quasar_fds.cpp.
// The count read back while the enable mask was empty; anything but zero is the failure.
constexpr uint32_t kSlotCountUnderEmptyEnable = 1;
// The group status observed after the input registers were cleared under held senders.
constexpr uint32_t kSlotStatusAfterClear = 2;
// The group 0 status, recorded when either group 0 check fails.
constexpr uint32_t kSlotIdleStatus = 3;
constexpr uint32_t kNumSlots = 4;
constexpr uint32_t kTimeoutStatus = 0x5A5A0005;
constexpr uint32_t kCountedWithoutEnable = 0x5A5A0006;
constexpr uint32_t kTimeoutCount = 0x5A5A0007;
constexpr uint32_t kTimeoutClear = 0x5A5A0008;
constexpr uint32_t kRelatchedAfterClear = 0x5A5A0009;
constexpr uint32_t kBadIdleMap = 0x5A5A000A;
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
