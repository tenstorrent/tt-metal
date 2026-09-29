// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

constexpr uint32_t kNumSlots = 1;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, 1);

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, kNumWorkers, poll_iterations)) {
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

    uint32_t done_count = 0;
    fds_kernel::wait_group_count(group_id, 1, poll_iterations, done_count);

    // Dropping the go acknowledges the done; only then may the worker disable its queue, because
    // disabling switches its wire back to the register path and would retract an uncounted done.
    overlay::FdsDispatch::fds_clear_go();

    fds_kernel::finish(status, l1_address, kNumSlots, (done_count != 0) ? kComplete : kTimeout);
}
