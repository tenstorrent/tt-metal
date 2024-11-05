// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/global_semaphore.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

GlobalSemaphore::GlobalSemaphore(Device *device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type) :
    device_(device), cores_(cores), initial_value_(initial_value) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global semaphore can only be created for L1 buffer types");
    TT_FATAL(device != nullptr, "Device cannot be null");
    TT_FATAL(cores.num_cores() > 0, "CoreRangeSet must have at least one core");
    auto device_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores = cores.num_cores();
    auto shard_parameters = ShardSpecBuffer(cores, {1, 1}, ShardOrientation::ROW_MAJOR, false, {1, 1}, {num_cores, 1});

    ShardedBufferConfig config{
        .device = device,
        .size = num_cores * sizeof(uint32_t),
        .page_size = sizeof(uint32_t),
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_parameters};
    this->buffer_ = Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        config.shard_parameters,
        std::nullopt);

    this->host_buffer_ = std::vector<uint32_t>(num_cores, initial_value);
    this->initialize();
}

std::shared_ptr<GlobalSemaphore> GlobalSemaphore::create(
    Device *device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type) {
    return std::make_shared<GlobalSemaphore>(device, cores, initial_value, buffer_type);
}

DeviceAddr GlobalSemaphore::address() const { return buffer_->address(); }

void GlobalSemaphore::initialize() {
    // Blocking write of semaphore value to buffer
    if (this->device_->using_slow_dispatch()) {
        detail::WriteToBuffer(*this->buffer_, this->host_buffer_);
        tt::Cluster::instance().l1_barrier(this->device_->id());
    } else {
        EnqueueWriteBuffer(this->device_->command_queue(), this->buffer_, this->host_buffer_, true);
    }
}

}  // namespace tt::tt_metal
