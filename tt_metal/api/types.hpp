// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {
inline namespace v0 {

class CommandQueue;
class Device;

}  // namespace v0

namespace v1 {

struct DeviceHandle {
    uint16_t key = 0;

    bool is_valid() const { return key != 0; }
};

struct CommandQueueHandle {
    public:
    operator v0::CommandQueue&() const { return command_queue; }

    private:
    friend CommandQueueHandle GetCommandQueue(DeviceHandle, uint32_t);
    CommandQueueHandle(v0::CommandQueue& queue) : command_queue(queue) {}

    v0::CommandQueue& command_queue;
};

}  // namespace v1

}  // namespace tt::tt_metal
