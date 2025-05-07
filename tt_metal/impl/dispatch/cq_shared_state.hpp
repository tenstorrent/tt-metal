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
    void TakeOwnership(SubDeviceId id, uint32_t cq_id) {
        if (this->cq_id != std::nullopt && this->cq_id != cq_id) {
            TT_FATAL(
                this->cq_id == cq_id,
                "Sub device id {} currently in use by cq {}. Can't enqueue program from cq {}. Finish or wait "
                "for an event to transfer ownership.",
                *id,
                *this->cq_id,
                cq_id);
        }
        this->cq_id = cq_id;
        ownership_event_id = std::nullopt;
    }

    void Finished(uint32_t cq_id) {
        if (this->cq_id.has_value() && this->cq_id == cq_id) {
            this->cq_id = std::nullopt;
            this->ownership_event_id = std::nullopt;
        }
    }

    void RecordedEvent(uint32_t event_id, uint32_t event_cq) {
        if (cq_id.has_value() && cq_id == event_cq) {
            if (ownership_event_id.has_value()) {
                TT_ASSERT(*ownership_event_id < event_id, "Ownership event ID must be less than the current event ID");
            } else {
                ownership_event_id = event_id;
            }
        }
    }

    void WaitedForEvent(uint32_t event_id, uint32_t event_cq, uint32_t cq_id) {
        if (this->cq_id.has_value() && event_cq == this->cq_id && this->cq_id != cq_id &&
            ownership_event_id.has_value() && ownership_event_id <= event_id) {
            this->cq_id = std::nullopt;
            ownership_event_id = std::nullopt;
        }
    }

private:
    std::optional<uint32_t> cq_id;               // The command queue ID that owns this sub-device.
    std::optional<uint32_t> ownership_event_id;  // The first event ID to wait on to grant ownership.
};

// State that is shared across all command queues for a device.
struct CQSharedState {
    DispatchArray<LaunchMessageRingBufferState> worker_launch_message_buffer_state;

    std::vector<CQOwnerState> sub_device_cq_owner;
};

}  // namespace tt::tt_metal
