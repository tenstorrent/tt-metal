// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the concurrent-engines test: every dispatch engine sends a go for the same group
// at once, and this kernel waits until its group status shows all of them held simultaneously
// before answering with a single done — which each engine then sees on its own lane. This is the
// only kernel that exercises the worker-side group decode across its input lanes; the standard
// worker polls raw registers with a threshold of one. The dispatch-engine side is the standard
// quasar_dispatch_engine_signal.cpp, one instance per engine in one program.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

// The last group status observed, so a timeout names which lanes were still missing.
constexpr uint32_t kSlotStatusMask = 1;
constexpr uint32_t kNumSlots = 2;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t num_engines = get_named_compile_time_arg_val("num_engines");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, kNumSlots);

    fds_epoch::clear_worker_inputs(dispatch_mask);

    // This is the one kernel whose opening wait is not fds_epoch::wait_for_go: instead of the first
    // go on any lane, it needs every engine's go held at the same time, read through the group
    // decode rather than the raw lane registers. The ready side is the shared protocol's — one
    // alternating token on this worker's single done wire serves every engine at once. Each engine
    // holds its go until its own done count is satisfied, which cannot happen before this kernel
    // answers, so requiring simultaneity cannot deadlock.
    //
    // No group configuration precedes the wait, because none would reach it: the enable mask gates
    // counting only and never status, so the decode read below needs nothing programmed. Lane
    // selection is done here in software, against dispatch_mask.
    uint32_t status_mask = 0;
    bool all_gos_held = false;
    uint32_t ready_token = kReadyTokenA;
    for (uint32_t i = 0; i < poll_iterations && !all_gos_held; i++) {
        fds_epoch::pulse_ready(ready_token, i);
        status_mask = overlay::FdsNeo::fds_read_group_status(group_id) & dispatch_mask;
        all_gos_held = static_cast<uint32_t>(__builtin_popcount(status_mask)) >= num_engines;
    }

    status[kSlotStatusMask] = status_mask;
    fds_kernel::finish(status, l1_address, kNumSlots, all_gos_held ? kComplete : kTimeout);

    if (all_gos_held) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);
    }
}
