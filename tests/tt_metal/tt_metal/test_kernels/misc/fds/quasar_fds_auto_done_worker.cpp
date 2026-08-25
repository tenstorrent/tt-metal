// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

constexpr uint32_t kSlotDoneReadback = 1;
constexpr uint32_t kNumSlots = 2;
constexpr uint32_t kTimeoutAck = kTimeoutGoClear;
constexpr uint32_t kNotDiverted = 0x5A5A0040;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t auto_dispatch_cycles = get_named_compile_time_arg_val("auto_dispatch_cycles");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kReadyTokenA, "payload group ids must stay below the ready tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, kNumSlots);

    uint32_t go_inst = 0;
    if (!fds_kernel::received_go(status, l1_address, kNumSlots, dispatch_mask, group_id, poll_iterations, go_inst)) {
        return;
    }

    // Zero the output register before switching modes: it is the value the wire falls back to when
    // the queue is disabled again, and the stale-readback check below needs a known value in it.
    overlay::FdsNeo::fds_clear_done();
    overlay::FdsNeo::fds_config_auto_dispatch(
        /*enable=*/true, auto_dispatch_cycles, TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR);

    overlay::FdsNeo::fds_done(/*ad_enable=*/true, group_id);

    // The queue path never touches the register, so this reads the zero from before the enable. A
    // readback of the group id means the write took the direct path.
    const uint32_t readback = overlay::FdsNeo::fds_read_done();
    status[kSlotDoneReadback] = readback;

    uint32_t result = kComplete;
    if (readback == group_id) {
        result = kNotDiverted;
    }

    if (result == kComplete) {
        // The engine drops the go once it has counted the done; disabling the queue any earlier
        // would retract the done, since the wire falls back to the (zeroed) register.
        if (!fds_kernel::wait_de_status(go_inst, 0, poll_iterations)) {
            result = kTimeoutAck;
        }
    }

    // Leave the outbox on the output bus rather than zeroing it: zero is the address of input
    // register 0, so a later enable that ran before reprogramming the outbox would divert a
    // go-clearing write into the queue and emit it as an outgoing done.
    overlay::FdsNeo::fds_config_auto_dispatch(
        /*enable=*/false, 0, TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR);

    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
