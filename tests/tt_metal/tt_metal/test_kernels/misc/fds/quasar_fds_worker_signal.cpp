// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Which dispatch instance drives this NEO is not established, so every inbox register is watched
// rather than a chosen one.
//
// The opening ready handshake, and why it is needed at all, is described in quasar_fds_epoch.h.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

constexpr uint32_t kNumSlots = 1;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, kNumSlots);

    uint32_t go_inst = 0;
    const bool go_received = fds_epoch::wait_for_go(dispatch_mask, group_id, poll_iterations, go_inst);

    // The real completion path does need its data ordered before the done, but this is not a test of
    // that: this is a local store that no reader consumes on seeing the done.
    fds_kernel::finish(status, l1_address, kNumSlots, go_received ? kComplete : kTimeout);

    if (go_received) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
    }
}
