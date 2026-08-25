// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the capture-semantics test, holding the assertions:
//  1. A software clear of an input register sticks while the sender holds its value: the register
//     must stay zero, because capture is change-triggered and the wire has not changed.
//  2. A rewrite of the identical value is not recaptured: writing the same group id again puts no
//     change on the wire, so exactly one capture happens per stable-value episode.
//  3. A real change through zero is recaptured, which also proves the silence above was capture
//     semantics and not a dead wire.
// The dispatch-engine side that paces these steps lives in quasar_fds_capture_dispatch.cpp.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_capture::kTokenChecked;
using fds_capture::kTokenCleared;

constexpr uint32_t kNumSlots = 2;
// The cleared input register re-latched the held value.
constexpr uint32_t kRelatchedAfterClear = 0x5A5A0010;
// The rewrite of the identical value was captured as a new event.
constexpr uint32_t kRecapturedWithoutChange = 0x5A5A0011;
// The real change through zero was never captured.
constexpr uint32_t kTimeoutRecapture = 0x5A5A0012;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t silence_iterations = get_named_compile_time_arg_val("silence_iterations");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    static_assert(group_id < kTokenCleared, "the payload group must not collide with the step tokens");

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, kNumSlots);

    uint32_t go_inst = 0;
    if (!fds_kernel::received_go(status, l1_address, kNumSlots, dispatch_mask, group_id, poll_iterations, go_inst)) {
        return;
    }

    uint32_t result = kComplete;
    uint32_t observed = 0;

    // The engine holds the go throughout this window, so anything but zero here is a re-latch.
    overlay::FdsNeo::fds_clear_de_status(go_inst);
    observed = fds_kernel::lane_nonzero(go_inst, silence_iterations);
    if (observed != 0) {
        result = kRelatchedAfterClear;
    }

    if (result == kComplete) {
        // The engine rewrites the identical value early in this window; with no change on the
        // wire, nothing may be captured.
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenCleared);
        observed = fds_kernel::lane_nonzero(go_inst, silence_iterations);
        if (observed != 0) {
            result = kRecapturedWithoutChange;
        }
    }

    if (result == kComplete) {
        // A rewrite that a wrong model delivered only after the window above closed would sit on
        // the lane and pre-satisfy the recapture wait below, so the lane must still be clear.
        observed = overlay::FdsNeo::fds_read_de_status(go_inst);
        if (observed != 0) {
            result = kRecapturedWithoutChange;
        }
    }

    if (result == kComplete) {
        // The engine now sends the value through zero: a real change, which must be captured.
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenChecked);
        if (!fds_kernel::wait_de_status(go_inst, group_id, poll_iterations)) {
            result = kTimeoutRecapture;
        }
    }

    status[kSlotObservedValue] = observed;
    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
