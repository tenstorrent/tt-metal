// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <optional>
#include <api/tt-metalium/sub_device_types.hpp>

#include "launch_message_ring_buffer_state.hpp"
#include "dispatch_settings.hpp"

namespace tt::tt_metal {

// Keeps track of the ownership state of a sub-device's workers.
class CQOwnerState {
public:
    // Raises an exception if the sub-device is already owned by a different command queue.
    void take_ownership(SubDeviceId id, uint32_t cq_id);

    void finished(uint32_t cq_id);
    void recorded_event(uint32_t event_id, uint32_t event_cq);
    void waited_for_event(uint32_t event_id, uint32_t event_cq, uint32_t cq_id);

    // Done-return address (go_msg_t word 1) ownership. The dispatcher coords written into the workers' go messages
    // belong to whichever CQ last reconfigured them; a CQ must re-emit CQ_DISPATCH_SET_GO_SIGNAL_NOC_ADDR before its
    // first go whenever word 1 does not already hold its coords (different CQ, or never set).
    bool needs_noc_addr_reconfigure(uint32_t cq_id) const { return noc_addr_cq_id_ != cq_id; }
    void mark_noc_addr_configured(uint32_t cq_id) { noc_addr_cq_id_ = cq_id; }

    // Host mirror of the dispatcher's per-sub-device GO counter (go_count_per_sync), shared across CQs because the
    // worker's go_processed is shared. It must advance in lockstep with every dispatcher go-signal for this sub-device
    // so that, on a CQ-ownership change, the reconfigure command can reseed the new owner's counter to the workers'
    // current go_processed. go_count() is that reseed baseline.
    uint8_t go_count() const { return go_count_; }
    void advance_go_count() { ++go_count_; }       // a program / replay go: dispatcher does go_count_per_sync++
    void rebaseline_go_count() { go_count_ = 1; }  // a RESET_READ_PTR go: dispatcher does (0 then ++) -> 1

private:
    std::optional<uint32_t> cq_id_;               // The command queue ID that owns this sub-device.
    std::optional<uint32_t> ownership_event_id_;  // The first event ID to wait on to grant ownership.
    std::optional<uint32_t> noc_addr_cq_id_;      // CQ whose dispatcher coords are currently in the workers' word 1.
    uint8_t go_count_ = 0;                        // mirrors the dispatcher's go_count_per_sync for this sub-device.
};

// State that is shared across all command queues for a device.
struct CQSharedState {
    DispatchArray<LaunchMessageRingBufferState> worker_launch_message_buffer_state;

    // One entry per sub-device.
    std::vector<CQOwnerState> sub_device_cq_owner;
};

}  // namespace tt::tt_metal
