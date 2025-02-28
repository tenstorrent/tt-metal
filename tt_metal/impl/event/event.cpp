// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <event.hpp>

#include <thread>

#include <assert.hpp>
#include <logger.hpp>
#include "tt_metal/event.hpp"

namespace tt::tt_metal {

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
