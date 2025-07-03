// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.hpp>
#include <event.hpp>
#include <tt-logger/tt-logger.hpp>
#include <chrono>
#include <thread>

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

}  // namespace tt::tt_metal
