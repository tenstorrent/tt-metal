// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <optional>
#include <api/tt-metalium/event.hpp>
#include <api/tt-metalium/sub_device_types.hpp>

#include "assert.hpp"
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

private:
    std::optional<uint32_t> cq_id_;               // The command queue ID that owns this sub-device.
    std::optional<uint32_t> ownership_event_id_;  // The first event ID to wait on to grant ownership.
};

// State that is shared across all command queues for a device.
struct CQSharedState {
    DispatchArray<LaunchMessageRingBufferState> worker_launch_message_buffer_state;

    // One entry per sub-device.
    std::vector<CQOwnerState> sub_device_cq_owner;
};

}  // namespace tt::tt_metal
