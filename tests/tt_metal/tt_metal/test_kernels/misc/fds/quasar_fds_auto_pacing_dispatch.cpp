// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The burst is written through fds_go with auto dispatch enabled, which polls the queue-full flag
// before each write; the flag is sampled once more before the last write, where four undelivered
// values must be sitting in the four-deep queue.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_auto_pacing::kBurstValueBase;
using fds_auto_pacing::kTokenArmed;
using fds_auto_pacing::kTokenRecorded;

constexpr uint32_t kSlotSawQueueFull = 1;
constexpr uint32_t kNumSlots = 2;
constexpr uint32_t kTimeoutArmed = 0x5A5A0006;
constexpr uint32_t kTimeoutRecorded = 0x5A5A0007;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t burst_length = get_named_compile_time_arg_val("burst_length");
    constexpr uint32_t auto_dispatch_cycles = get_named_compile_time_arg_val("auto_dispatch_cycles");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(kBurstValueBase + burst_length <= kTokenRecorded, "the burst must stay below the step tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    overlay::FdsDispatch::fds_config_groupid(kTokenArmed, worker_mask, 1);
    overlay::FdsDispatch::fds_config_groupid(kTokenRecorded, worker_mask, 1);

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, kNumWorkers, poll_iterations)) {
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kSessionGo);

    if (!fds_kernel::wait_group_count_nonzero(kTokenArmed, poll_iterations)) {
        fds_kernel::finish(status, l1_address, kNumSlots, kTimeoutArmed);
        return;
    }

    overlay::FdsDispatch::fds_config_auto_dispatch_pacing(auto_dispatch_cycles);
    overlay::FdsDispatch::fds_config_auto_dispatch_outbox(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR);
    overlay::FdsDispatch::fds_enable_auto_dispatch();

    for (uint32_t i = 0; i < burst_length; i++) {
        if (i == burst_length - 1) {
            status[kSlotSawQueueFull] = overlay::FdsDispatch::fds_read_auto_dispatch_fifo_full();
        }
        overlay::FdsDispatch::fds_go(/*ad_enable=*/true, kBurstValueBase + i);
    }

    const bool recorded = fds_kernel::wait_group_count_nonzero(kTokenRecorded, poll_iterations);

    // Back to the direct path: the wire falls back to the output register, which still holds the
    // session go, so clear it once the direct path is active again. The pacing is left alone: the
    // counter is mid-interval after the burst, and a cycle count it has already passed would
    // strand the queue until a 32 bit wrap.
    overlay::FdsDispatch::fds_disable_auto_dispatch();
    overlay::FdsDispatch::fds_clear_go();

    fds_kernel::finish(status, l1_address, kNumSlots, recorded ? kComplete : kTimeoutRecorded);
}
