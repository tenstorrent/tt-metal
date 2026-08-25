// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the auto dispatch pacing test: record every value the engine's queue releases
// onto the go wire, in the order captured. The burst counts upwards, so each release is a change
// the filter passes and each value is unique — which is what lets the host check order and
// multiplicity rather than merely counting transitions. The dispatch-engine side lives in
// quasar_fds_auto_pacing_dispatch.cpp.
//
// One limit is the hardware's, not the test's: a release of the value already on the wire is no
// change at all, so nothing downstream could observe it. Unique burst values sidestep that instead
// of trying to detect it.
//
// The release cadence must comfortably exceed this kernel's polling period, or values would be
// overwritten between reads; the host picks the cadence with that in mind.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_auto_pacing::kTokenArmed;
using fds_auto_pacing::kTokenRecorded;

constexpr uint32_t kSlotRecordedCount = 1;
constexpr uint32_t kSlotFirstValue = 2;
// Fewer than burst_length values arrived.
constexpr uint32_t kTimeoutBurst = 0x5A5A0050;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t burst_length = get_named_compile_time_arg_val("burst_length");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    constexpr uint32_t num_slots = kSlotFirstValue + burst_length;

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, num_slots);

    uint32_t go_inst = 0;
    if (!fds_kernel::received_go(status, l1_address, num_slots, dispatch_mask, kSessionGo, poll_iterations, go_inst)) {
        return;
    }

    // Clear the lane so the burst starts from a quiet register, then tell the engine to fire.
    overlay::FdsNeo::fds_clear_de_status(go_inst);
    overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenArmed);

    // Every burst value is distinct, so a new capture is simply a new register value. Zero is
    // skipped because the mode switch to auto dispatch may put an idle zero on the wire, and the
    // session go value is skipped in case that switch briefly re-presents it; the burst
    // deliberately uses neither.
    uint32_t recorded = 0;
    uint32_t last_value = 0;
    for (uint32_t i = 0; i < poll_iterations && recorded < burst_length; i++) {
        const uint32_t value = overlay::FdsNeo::fds_read_de_status(go_inst);
        if (value != 0 && value != kSessionGo && value != last_value) {
            status[kSlotFirstValue + recorded] = value;
            recorded++;
            last_value = value;
        }
    }

    status[kSlotRecordedCount] = recorded;
    fds_kernel::finish(status, l1_address, num_slots, (recorded == burst_length) ? kComplete : kTimeoutBurst);

    if (recorded == burst_length) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenRecorded);
    }
}
