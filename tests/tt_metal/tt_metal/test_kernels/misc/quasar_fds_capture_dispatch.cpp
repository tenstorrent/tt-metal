// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the capture-semantics test. Capture is change-triggered: a software
// clear of an input register sticks while the sender still holds its value, and rewriting the same
// value produces no second capture because the wire never changes. This kernel supplies the held,
// rewritten and changed go values in turn, paced by tokens the worker raises on its done wire; the
// checks live in quasar_fds_capture_worker.cpp.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_capture::kTokenChecked;
using fds_capture::kTokenCleared;

constexpr uint32_t kNumSlots = 1;
// The worker never reported clearing its input register.
constexpr uint32_t kTimeoutCleared = 0x5A5A0006;
// The worker never reported finishing its rewrite-silence window.
constexpr uint32_t kTimeoutChecked = 0x5A5A0007;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kTokenCleared, "the payload group must not collide with the step tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    overlay::FdsDispatch::fds_config_groupid(kTokenCleared, worker_mask, 1);
    overlay::FdsDispatch::fds_config_groupid(kTokenChecked, worker_mask, 1);

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, kNumWorkers, poll_iterations)) {
        return;
    }

    // The held value. The worker will observe it, clear its input register, and verify the clear
    // sticks while this value stays on the wire.
    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

    uint32_t result = kComplete;
    if (!fds_kernel::wait_group_count_nonzero(kTokenCleared, poll_iterations)) {
        result = kTimeoutCleared;
    }

    if (result == kComplete) {
        // Rewrite of the identical value: the register and the wire never change, so the worker
        // must see no second capture.
        overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);
        if (!fds_kernel::wait_group_count_nonzero(kTokenChecked, poll_iterations)) {
            result = kTimeoutChecked;
        }
    }

    if (result == kComplete) {
        // A real change: through zero and back. Only this may recapture at the worker.
        overlay::FdsDispatch::fds_clear_go();
        overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);
    }

    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
