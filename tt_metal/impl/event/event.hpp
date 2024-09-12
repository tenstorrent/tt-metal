// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <thread>
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/logger.hpp"
namespace tt::tt_metal
{
    class Device;
    struct Event
    {
        Device * device = nullptr;
        uint32_t cq_id = -1;
        uint32_t event_id = -1;
        std::atomic<bool> ready = false; // Event is ready for use.

        // With async CQ, must wait until event is populated by child thread before using.
        // Opened #5988 to track removing this, and finding different solution.
        void wait_until_ready() {
            while (!ready) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                log_trace(tt::LogMetal, "Waiting for Event to be ready. (ready: {} cq_id: {} event_id: {})", bool(ready), cq_id, event_id);
            }

            TT_ASSERT(device != nullptr, "Event must have initialized device ptr");
            TT_ASSERT(event_id != -1, "Event must have initialized event_id");
            TT_ASSERT(cq_id != -1, "Event must have initialized cq_id");
        }
    };
}
