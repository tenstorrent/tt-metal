// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the de-glitch filter test: supply a sub-threshold pulse, a held value,
// and a held value at the specification's floor threshold, each on the worker's cue. The
// assertions live in quasar_fds_filter_worker.cpp.
//
// The session go both starts the worker and tells it which lane this engine drives. It is dropped
// again before the worker raises its filter: a threshold raised over a parked lane re-captures the
// value still on the wire, so the wire must be carrying zero by then for that re-capture to be
// invisible.

#include <cstdint>
#include "api/compile_time_args.h"

#include "quasar_fds_common.h"

using fds_filter::kPayloadGo;
using fds_filter::kStepTokens;
using fds_filter::kTokenArmed;
using fds_filter::kTokenDone;
using fds_filter::kTokenLaneKnown;
using fds_filter::kTokenPulseChecked;
using fds_filter::kTokenRearmed;

constexpr uint32_t kNumSlots = 1;
// A step token never arrived; the value encodes which one below.
constexpr uint32_t kTimeoutTokenBase = 0x5A5A0020;
constexpr uint32_t kPulseRepeats = 8;

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    fds_kernel::status_ptr status = fds_kernel::begin_dispatch(l1_address, kNumSlots);
    for (const uint32_t token : kStepTokens) {
        overlay::FdsDispatch::fds_config_groupid(token, worker_mask, 1);
    }

    if (!fds_kernel::workers_are_ready(status, l1_address, kNumSlots, worker_mask, kNumWorkers, poll_iterations)) {
        return;
    }

    overlay::FdsDispatch::fds_clear_go();
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kSessionGo);

    uint32_t result = kComplete;
    constexpr uint32_t num_steps = sizeof(kStepTokens) / sizeof(kStepTokens[0]);
    for (uint32_t step = 0; step < num_steps && result == kComplete; step++) {
        if (!fds_kernel::wait_group_count_nonzero(kStepTokens[step], poll_iterations)) {
            result = kTimeoutTokenBase + step;
            break;
        }
        // Keyed on the token rather than the loop ordinal, so each action names the cue it answers
        // and reordering the step list cannot silently rebind the actions below it.
        switch (kStepTokens[step]) {
            case kTokenLaneKnown:
                // Drop the session go so the worker sees the zero land before raising its filter.
                overlay::FdsDispatch::fds_clear_go();
                break;
            case kTokenArmed:
                // The pulses: each pair of adjacent writes fits the 2-deep outbound crossing, and
                // the read between pairs lets it drain. Every payload lives on the wire far shorter
                // than the worker's long filter threshold, so all of them must be lost. One pulse
                // would prove the filter; the repetition is for a filterless model, whose brief
                // capture of a payload could fall between two of the worker's polls.
                for (uint32_t i = 0; i < kPulseRepeats; i++) {
                    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kPayloadGo);
                    overlay::FdsDispatch::fds_clear_go();
                    overlay::FdsDispatch::fds_read_group_count(kTokenArmed);
                }
                break;
            case kTokenPulseChecked:
                // The held value: stable until the worker reports capture, so the long filter must
                // pass it.
                overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kPayloadGo);
                break;
            case kTokenRearmed:
                // A fresh change for the floor-threshold capture.
                overlay::FdsDispatch::fds_clear_go();
                overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kPayloadGo);
                break;
            case kTokenDone: overlay::FdsDispatch::fds_clear_go(); break;
        }
    }

    fds_kernel::finish(status, l1_address, kNumSlots, result);
}
