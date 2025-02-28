// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>

namespace tt::tt_metal {
inline namespace v0 {

class IDevice;

struct Event {
    IDevice* device = nullptr;
    uint32_t cq_id = -1;
    uint32_t event_id = -1;
};

}  // namespace v0
}  // namespace tt::tt_metal
