// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

inline namespace v0 {
class Buffer;
class Device;
}  // namespace v0

class GlobalSemaphore {
   public:
    GlobalSemaphore(Device *device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    static std::shared_ptr<GlobalSemaphore> create(
        Device *device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    DeviceAddr address() const;

    void initialize();

   private:
    // GlobalSemaphore is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    std::shared_ptr<Buffer> buffer_;
    std::vector<uint32_t> host_buffer_;
    Device *device_;
    CoreRangeSet cores_;
    uint32_t initial_value_;
};

}  // namespace tt::tt_metal
