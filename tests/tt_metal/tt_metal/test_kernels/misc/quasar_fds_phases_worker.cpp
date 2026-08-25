// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the consecutive-phases test: answer several go/done rounds on the same group,
// clearing the done between rounds as the protocol requires for identical messages. The
// dispatch-engine side lives in quasar_fds_phases_dispatch.cpp.
//
// The go dropping to zero is the signal that the engine has collected this round's dones, so the
// done is cleared only then: clearing earlier would retract it before it was counted.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

constexpr uint32_t kSlotPhasesDone = 1;
constexpr uint32_t kNumSlots = 2;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t num_phases = get_named_compile_time_arg_val("num_phases");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, kNumSlots);

    // The engine is fixed for the whole run, so later phases watch the lane the first go arrived on
    // and nothing else.
    uint32_t go_inst = 0;
    if (!fds_kernel::received_go(status, l1_address, kNumSlots, dispatch_mask, group_id, poll_iterations, go_inst)) {
        return;
    }

    uint32_t phases_done = 0;
    uint32_t result = kComplete;
    for (uint32_t phase = 0; phase < num_phases; phase++) {
        if (phase > 0 && !fds_kernel::wait_de_status(go_inst, group_id, poll_iterations)) {
            result = kTimeoutGo;
            break;
        }

        overlay::FdsNeo::fds_done(/*ad_enable=*/false, group_id);

        if (!fds_kernel::wait_de_status(go_inst, 0, poll_iterations)) {
            result = kTimeoutGoClear;
            break;
        }

        overlay::FdsNeo::fds_clear_done();
        phases_done++;
    }

    status[kSlotPhasesDone] = phases_done;
    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
