// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Worker side of the de-glitch filter test, holding the assertions:
//  1. A value replaced before the filter threshold elapses is genuinely lost: after the engine's
//     brief pulses, nothing may be captured for the whole silence window.
//  2. A value held stable is captured, at the same long threshold that filtered the pulse.
//  3. The specification's floor threshold of 7 also passes a held value.
// The dispatch-engine side that supplies pulse and held values lives in
// quasar_fds_filter_dispatch.cpp.
//
// Two threshold hazards shape the sequencing. Raising the threshold over a parked lane re-captures
// whatever the wire holds, so the long filter is programmed only after the engine has dropped the
// session go and its zero has demonstrably landed — the re-capture is then a harmless zero.
// Lowering it below a parked count only produces a duplicate a wraparound away, which no test run
// lives to see. Filter writes also cross a clock domain before the filter sees them; every
// reprogramming here is separated from the traffic it must govern by a token round trip, which
// dwarfs that delay.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_filter::kPayloadGo;
using fds_filter::kTokenArmed;
using fds_filter::kTokenDone;
using fds_filter::kTokenLaneKnown;
using fds_filter::kTokenPulseChecked;
using fds_filter::kTokenRearmed;

constexpr uint32_t kNumSlots = 2;
// The sub-threshold pulse was captured.
constexpr uint32_t kPulseCaptured = 0x5A5A0030;
// The held value was never captured at the long threshold.
constexpr uint32_t kTimeoutHeldCapture = 0x5A5A0031;
// The held value was never captured at the floor threshold.
constexpr uint32_t kTimeoutFloorCapture = 0x5A5A0032;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t dispatch_mask = get_named_compile_time_arg_val("dispatch_mask");
    constexpr uint32_t long_filter = get_named_compile_time_arg_val("long_filter");
    constexpr uint32_t floor_filter = get_named_compile_time_arg_val("floor_filter");
    constexpr uint32_t silence_iterations = get_named_compile_time_arg_val("silence_iterations");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    fds_kernel::status_ptr status = fds_kernel::begin_worker(l1_address, kNumSlots);

    uint32_t go_inst = 0;
    if (!fds_kernel::received_go(status, l1_address, kNumSlots, dispatch_mask, kSessionGo, poll_iterations, go_inst)) {
        return;
    }

    uint32_t result = kComplete;
    uint32_t observed = 0;

    // Ask for the session go to be dropped, and let the hardware capture of its zero prove the
    // wire is clear before the filter is raised.
    overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenLaneKnown);
    if (!fds_kernel::wait_de_status(go_inst, 0, poll_iterations)) {
        result = kTimeoutGoClear;
    }

    if (result == kComplete) {
        overlay::FdsNeo::fds_config_filter_length(long_filter);
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenArmed);

        // The engine fires a train of brief pulses early in this window, each far shorter than
        // the long threshold, so the lane must stay clear.
        observed = fds_kernel::lane_nonzero(go_inst, silence_iterations);
        if (observed != 0) {
            result = kPulseCaptured;
        }
    }

    if (result == kComplete) {
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenPulseChecked);
        if (!fds_kernel::wait_de_status(go_inst, kPayloadGo, poll_iterations)) {
            result = kTimeoutHeldCapture;
        }
    }

    if (result == kComplete) {
        overlay::FdsNeo::fds_clear_de_status(go_inst);
        overlay::FdsNeo::fds_config_filter_length(floor_filter);
        overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenRearmed);
        if (!fds_kernel::wait_de_status(go_inst, kPayloadGo, poll_iterations)) {
            result = kTimeoutFloorCapture;
        } else {
            overlay::FdsNeo::fds_done(/*ad_enable=*/false, kTokenDone);
        }
    }

    // Leave de-glitching off for whatever runs on this tile next.
    overlay::FdsNeo::fds_config_filter_length(kNoDeglitchFilter);

    status[kSlotObservedValue] = observed;
    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
