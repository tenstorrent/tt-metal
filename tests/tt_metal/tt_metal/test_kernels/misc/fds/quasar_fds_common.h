// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared status layout, bounded waits, and pair-specific tokens for the Quasar FDS test kernels.

#include "risc_attribs.h"
#include "risc_common.h"

#include "quasar_fds_epoch.h"

// Slot 0 of every status block. Extra slots are per-kernel and documented next to the kernel that
// writes them.
constexpr uint32_t kSlotResult = 0;
// Workers that record a value that broke a silence window put it here.
constexpr uint32_t kSlotObservedValue = 1;

constexpr uint32_t kStarted = 0x5A5A0001;
constexpr uint32_t kComplete = 0x5A5A0002;
constexpr uint32_t kTimeout = 0x5A5A0003;
// The ready wait timed out: some worker never presented a ready token, so no go was ever sent.
constexpr uint32_t kReadyTimeout = 0x5A5A0004;
// Worker: the awaited go never arrived. Same value as kTimeout; the name is the worker-side
// reading of that failure.
constexpr uint32_t kTimeoutGo = kTimeout;
// Worker: a go that was seen never dropped back to zero.
constexpr uint32_t kTimeoutGoClear = 0x5A5A0005;

constexpr uint32_t kNumWorkers = 1;

// Opening value for the tests that need one distinct from their payload, so a stale session capture
// cannot satisfy a later payload wait. Only those tests are bound by it, and they pick payload
// values other than zero and this; a test with no session go uses 1 as an ordinary payload group
// id. Every payload id is bounded above by the ready tokens, which each kernel static_asserts.
constexpr uint32_t kSessionGo = 1;

namespace fds_kernel {

using status_ptr = volatile tt_l1_ptr uint32_t*;

inline status_ptr begin(uint32_t l1_address, uint32_t num_slots) {
    status_ptr status = reinterpret_cast<status_ptr>(l1_address);
    status[kSlotResult] = kStarted;
    flush_l2_cache_range(l1_address, num_slots * sizeof(uint32_t));
    return status;
}

inline void finish(status_ptr status, uint32_t l1_address, uint32_t num_slots, uint32_t result) {
    status[kSlotResult] = result;
    flush_l2_cache_range(l1_address, num_slots * sizeof(uint32_t));
}

inline status_ptr begin_dispatch(uint32_t l1_address, uint32_t num_slots) {
    status_ptr status = begin(l1_address, num_slots);
    // Auto dispatch is disabled defensively: a kernel that died between enabling it and its
    // teardown would leave the output multiplexer on the queue path, turning every later direct
    // write on this engine into silence far from the fault. The outbox parks on the output bus,
    // so that even a mistimed enable diverts only writes that were headed for the wire anyway.
    // This is everything software can reset. The pacing counter is unreachable and keeps running
    // across kernels, and one test process shares one device, so pacing state from an earlier
    // test is still live here. The pacing register is therefore left alone: writing it while the
    // counter is mid-interval strands the queue until a 32 bit wrap.
    overlay::FdsDispatch::fds_disable_auto_dispatch();
    overlay::FdsDispatch::fds_config_auto_dispatch_outbox(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR);
    overlay::FdsDispatch::fds_config_filter_length(kNoDeglitchFilter);
    return status;
}

inline status_ptr begin_worker(uint32_t l1_address, uint32_t num_slots) {
    status_ptr status = begin(l1_address, num_slots);
    // Same defensive disable as begin_dispatch, and the same limit on what it can reset. The
    // outbox park matters more on this map: zero is input register 0, and a stale zero outbox
    // under a mistimed enable would divert a status-clearing write into the queue and emit it as
    // an outgoing done.
    overlay::FdsNeo::fds_disable_auto_dispatch();
    overlay::FdsNeo::fds_config_auto_dispatch_outbox(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR);
    overlay::FdsNeo::fds_config_filter_length(kNoDeglitchFilter);
    return status;
}

inline bool workers_are_ready(
    status_ptr status,
    uint32_t l1_address,
    uint32_t num_slots,
    uint32_t worker_mask,
    uint32_t num_ready,
    uint32_t poll_iterations) {
    if (fds_epoch::wait_for_workers(worker_mask, num_ready, poll_iterations)) {
        return true;
    }
    finish(status, l1_address, num_slots, kReadyTimeout);
    return false;
}

inline bool received_go(
    status_ptr status,
    uint32_t l1_address,
    uint32_t num_slots,
    uint32_t dispatch_mask,
    uint32_t awaited_value,
    uint32_t poll_iterations,
    uint32_t& go_inst) {
    if (fds_epoch::wait_for_go(dispatch_mask, awaited_value, poll_iterations, go_inst)) {
        return true;
    }
    finish(status, l1_address, num_slots, kTimeoutGo);
    return false;
}

inline bool wait_group_count_nonzero(uint32_t group_id, uint32_t poll_iterations) {
    for (uint32_t i = 0; i < poll_iterations; i++) {
        if (overlay::FdsDispatch::fds_read_group_count(group_id) != 0) {
            return true;
        }
    }
    return false;
}

// Last observed count is left in `count`. True once it meets `threshold`.
inline bool wait_group_count(uint32_t group_id, uint32_t threshold, uint32_t poll_iterations, uint32_t& count) {
    count = 0;
    for (uint32_t i = 0; i < poll_iterations; i++) {
        count = overlay::FdsDispatch::fds_read_group_count(group_id);
        if (count >= threshold) {
            return true;
        }
    }
    return false;
}

inline bool wait_group_count_zero(uint32_t group_id, uint32_t poll_iterations, uint32_t& count) {
    for (uint32_t i = 0; i < poll_iterations; i++) {
        count = overlay::FdsDispatch::fds_read_group_count(group_id);
        if (count == 0) {
            return true;
        }
    }
    return false;
}

inline bool wait_de_status(uint32_t inst, uint32_t expected, uint32_t poll_iterations) {
    for (uint32_t i = 0; i < poll_iterations; i++) {
        if (overlay::FdsNeo::fds_read_de_status(inst) == expected) {
            return true;
        }
    }
    return false;
}

inline uint32_t lane_nonzero(uint32_t inst, uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        const uint32_t observed = overlay::FdsNeo::fds_read_de_status(inst);
        if (observed != 0) {
            return observed;
        }
    }
    return 0;
}

}  // namespace fds_kernel

// Step tokens and payload values shared by a dispatch/worker pair. Group ids 14 and 15 remain the
// ready tokens; these stay in 1..13.

namespace fds_filter {
constexpr uint32_t kTokenLaneKnown = 12;
constexpr uint32_t kTokenArmed = 8;
constexpr uint32_t kTokenPulseChecked = 9;
constexpr uint32_t kTokenRearmed = 10;
constexpr uint32_t kTokenDone = 11;
constexpr uint32_t kStepTokens[] = {kTokenLaneKnown, kTokenArmed, kTokenPulseChecked, kTokenRearmed, kTokenDone};
constexpr uint32_t kPayloadGo = 2;
}  // namespace fds_filter

namespace fds_capture {
constexpr uint32_t kTokenCleared = 8;
constexpr uint32_t kTokenChecked = 9;
}  // namespace fds_capture

namespace fds_auto_pacing {
constexpr uint32_t kTokenArmed = 9;
constexpr uint32_t kTokenRecorded = 8;
// The burst counts up from here, one value per release. Distinct values rather than two
// alternating ones are what make the host's per-index check an assertion: every release is still a
// change on the wire, but a repeat or a transposition now shows up as a wrong value at that index.
constexpr uint32_t kBurstValueBase = 2;
}  // namespace fds_auto_pacing

namespace fds_outbox {
constexpr uint32_t kTokenArmed = 9;
constexpr uint32_t kTokenSilenceChecked = 10;
constexpr uint32_t kTokenDelivered = 11;
constexpr uint32_t kMismatchedGo = 2;
constexpr uint32_t kMatchedGo = 3;
}  // namespace fds_outbox
