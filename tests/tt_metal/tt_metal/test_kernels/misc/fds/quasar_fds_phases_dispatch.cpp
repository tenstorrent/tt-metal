// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Phase boundaries ride on the wires themselves: the go dropping to zero tells the workers the
// engine has collected every done, and the group count falling back to zero tells the engine every
// worker has cleared its done and re-armed for the next phase.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

constexpr uint32_t kSlotPhasesDone = 1;
constexpr uint32_t kNumSlots = 2;
constexpr uint32_t kTimeoutDones = 0x5A5A0003;
constexpr uint32_t kTimeoutRearm = 0x5A5A0005;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t done_threshold = get_named_compile_time_arg_val("done_threshold");
    constexpr uint32_t num_phases = get_named_compile_time_arg_val("num_phases");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, done_threshold);

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, done_threshold, poll_iterations)) {
        return;
    }

    overlay::FdsDispatch::fds_clear_go();

    uint32_t phases_done = 0;
    uint32_t result = kComplete;
    for (uint32_t phase = 0; phase < num_phases; phase++) {
        overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

        uint32_t done_count = 0;
        if (!fds_kernel::wait_group_count(group_id, done_threshold, poll_iterations, done_count)) {
            result = kTimeoutDones;
            break;
        }

        // Dropping the go releases the workers to clear their dones; the count is live, so it
        // falls as each cleared done reaches this engine, and zero means every worker re-armed.
        overlay::FdsDispatch::fds_clear_go();
        if (!fds_kernel::wait_group_count_zero(group_id, poll_iterations, done_count)) {
            result = kTimeoutRearm;
            break;
        }
        phases_done++;
    }

    status[kSlotPhasesDone] = phases_done;
    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
