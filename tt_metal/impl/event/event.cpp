// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <event.hpp>

#include <thread>

#include <assert.hpp>
#include <logger.hpp>
#include "tt_metal/event.hpp"

namespace tt::tt_metal {

void Event::wait_until_ready() {
    while (!ready) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        log_trace(
            tt::LogMetal,
            "Waiting for Event to be ready. (ready: {} cq_id: {} event_id: {})",
            bool(ready),
            cq_id,
            event_id);
    }

    TT_ASSERT(device != nullptr, "Event must have initialized device ptr");
    TT_ASSERT(event_id != -1, "Event must have initialized event_id");
    TT_ASSERT(cq_id != -1, "Event must have initialized cq_id");
}

v1::EventHandle::EventHandle() : EventHandle(std::make_shared<Event>()) {}

v1::EventHandle v1::EnqueueRecordEvent(CommandQueueHandle cq) {
    EventHandle event{};
    v0::EnqueueRecordEvent(
        GetDevice(cq)->command_queue(GetId(cq)), static_cast<const std::shared_ptr<v0::Event>&>(event));
    return event;
}

void v1::EnqueueWaitForEvent(CommandQueueHandle cq, const EventHandle& event) {
    v0::EnqueueWaitForEvent(
        GetDevice(cq)->command_queue(GetId(cq)), static_cast<const std::shared_ptr<v0::Event>&>(event));
}

void v1::EventSynchronize(const EventHandle& event) {
    v0::EventSynchronize(static_cast<const std::shared_ptr<v0::Event>&>(event));
}

bool v1::EventQuery(const EventHandle& event) {
    return v0::EventQuery(static_cast<const std::shared_ptr<v0::Event>&>(event));
}

void v1::DeviceSynchronize(IDevice* device) { v0::Synchronize(device); }

void v1::CommandQueueSynchronize(CommandQueueHandle cq) { v0::Synchronize(GetDevice(cq), GetId(cq)); }

}  // namespace tt::tt_metal
