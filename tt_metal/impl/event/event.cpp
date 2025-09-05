// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.hpp>
#include <event.hpp>
#include <tt-logger/tt-logger.hpp>
#include <chrono>
#include <thread>

#include "device.hpp"
#include "impl/context/metal_context.hpp"

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

void Event::synchronize() {
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        // Slow dispatch conservatively flushes all work since there's no cq.
        device->synchronize();
        return;
    }
    wait_until_ready();  // Block until event populated. Parent thread.
    log_trace(
        tt::LogMetal,
        "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})",
        device->id(),
        cq_id,
        event_id);

    while (device->sysmem_manager().get_last_completed_event(cq_id) < event_id) {
        if (tt::tt_metal::MetalContext::instance().rtoptions().get_test_mode_enabled() &&
            MetalContext::instance().watcher_server()->killed_due_to_error()) {
            TT_FATAL(
                false,
                "Command Queue could not complete EventSynchronize. See {} for details.",
                MetalContext::instance().watcher_server()->log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

bool Event::query() {
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
        // Slow dispatch always returns true to avoid infinite blocking. Unclear if this is safe for all situations.
        return true;
    }
    wait_until_ready();  // Block until event populated. Parent thread.
    bool event_completed = device->sysmem_manager().get_last_completed_event(cq_id) >= event_id;
    log_trace(
        tt::LogMetal,
        "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed,
        device->id(),
        cq_id,
        event_id);
    return event_completed;
}

}  // namespace tt::tt_metal
