// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the Quasar FDS two-epoch re-arm experiment. The worker side lives in
// quasar_fds_worker_rearm.cpp.
//
// Both directions of the sideband are held levels rather than pulses, so a second done for a group
// is not distinguishable from the first one still being driven unless something de-asserts in
// between. This runs two epochs of the same group and measures what happens at each de-assert:
//
//   1. Send a go, wait for the worker's done.
//   2. With the worker still driving that same done, clear the receive inboxes and read the group
//      count immediately and again after a settle period. If the count returns to zero and stays
//      there, a sink-side clear holds against a live source. If it comes straight back, it does
//      not, and the simple protocol cannot tell two epochs of one group apart.
//   3. Drop the go to the idle group, wait, then re-assert it. The worker cannot tell epoch two
//      from epoch one without seeing the go de-assert, so this step is what makes the second epoch
//      addressable at all.
//   4. Wait for the worker's second done, which must appear as exactly one new credit.
//
// Every wait is bounded, and each measurement lands in its own status slot, so a protocol that
// fails part way through says where it stopped rather than hanging.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_signal_status.h"

namespace {

// One TENSIX_TO_DISPATCH inbox register per NEO wire on the dispatch side.
constexpr uint32_t kNumNeoWires = 32;

// Group 0 is the idle value on the wire, so writing it to the outbox is how a go is de-asserted.
constexpr uint32_t kIdleGroup = 0;

// A done must be stable for a single cycle to be accepted. This is the register's reset value,
// written explicitly so the handshake does not depend on the state a previous program left behind.
constexpr uint32_t kNoDeglitchFilter = 0;

void clear_all_inboxes() {
    for (uint32_t neo = 0; neo < kNumNeoWires; neo++) {
        overlay::FdsDispatch::fds_clear_neo_status(neo);
    }
}

// Bounded wait for the group count to reach the threshold, returning the last count seen.
uint32_t wait_for_done(uint32_t group_id, uint32_t threshold, uint32_t iterations) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < iterations; i++) {
        count = overlay::FdsDispatch::fds_read_group_count(group_id);
        if (count >= threshold) {
            break;
        }
    }
    return count;
}

void spin(uint32_t iterations) { for (volatile uint32_t i = 0; i < iterations; i++); }

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t group_id = get_named_compile_time_arg_val("group_id");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");
    // Long enough to outlast the deglitch interval, which is why a clear is read twice rather than
    // once: an immediate zero that does not survive a settle is not a clear that held.
    constexpr uint32_t settle_iterations = get_named_compile_time_arg_val("settle_iterations");

    // One worker takes part, so one lane can drive a done and the count tops out at one.
    constexpr uint32_t done_threshold = 1;

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::rearm::kSlotStarted] = quasar_fds_test::kStarted;
    status[quasar_fds_test::rearm::kSlotResult] = quasar_fds_test::kTimeout;
    flush_l2_cache_range(l1_address, quasar_fds_test::rearm::kNumSlots * sizeof(uint32_t));

    overlay::FdsDispatch::fds_config_filter_length(kNoDeglitchFilter);
    overlay::FdsDispatch::fds_config_groupid(group_id, worker_mask, done_threshold);
    clear_all_inboxes();

    // Epoch one.
    overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);
    const uint32_t round1_count = wait_for_done(group_id, done_threshold, poll_iterations);
    status[quasar_fds_test::rearm::kSlotRound1Count] = round1_count;

    if (round1_count >= done_threshold) {
        // The worker is still driving its done here: it only clears after seeing the go drop, and
        // the go is still held. So this measures a sink-side clear against a live source.
        clear_all_inboxes();
        status[quasar_fds_test::rearm::kSlotCountAfterClear] = overlay::FdsDispatch::fds_read_group_count(group_id);
        spin(settle_iterations);
        status[quasar_fds_test::rearm::kSlotCountAfterSettle] = overlay::FdsDispatch::fds_read_group_count(group_id);

        // De-assert the go, hold it idle long enough for the worker to notice, then re-assert. This
        // is what gives the worker an edge to distinguish the second epoch by.
        overlay::FdsDispatch::fds_go(/*ad_enable=*/false, kIdleGroup);
        spin(settle_iterations);
        overlay::FdsDispatch::fds_go(/*ad_enable=*/false, group_id);

        // Epoch two. The inboxes were cleared above, so this count starts from whatever survived
        // that clear rather than from zero by assumption.
        const uint32_t round2_count = wait_for_done(group_id, done_threshold, poll_iterations);
        status[quasar_fds_test::rearm::kSlotRound2Count] = round2_count;
        if (round2_count >= done_threshold) {
            status[quasar_fds_test::rearm::kSlotResult] = quasar_fds_test::kComplete;
        }
    }

    flush_l2_cache_range(l1_address, quasar_fds_test::rearm::kNumSlots * sizeof(uint32_t));
}
