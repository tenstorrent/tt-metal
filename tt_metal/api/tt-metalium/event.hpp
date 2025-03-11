// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>

namespace tt::tt_metal {

class IDevice;

struct Event {
    IDevice* device = nullptr;
    uint32_t cq_id = -1;
    uint32_t event_id = -1;
    std::atomic<bool> ready = false;  // Event is ready for use.

    // With async CQ, must wait until event is populated by child thread before using.
    // Opened #5988 to track removing this, and finding different solution.
    void wait_until_ready();
};

}  // namespace tt::tt_metal
