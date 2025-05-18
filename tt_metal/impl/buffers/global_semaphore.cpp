// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include <global_semaphore.hpp>
#include <host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt_metal.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) :
    device_(device), cores_(cores) {
    this->setup_buffer(initial_value, buffer_type);
}

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type) :
    device_(device), cores_(std::move(cores)) {
    this->setup_buffer(initial_value, buffer_type);
}

void GlobalSemaphore::setup_buffer(uint32_t initial_value, BufferType buffer_type) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global semaphore can only be created for L1 buffer types");
    TT_FATAL(device_ != nullptr, "Device cannot be null");
    TT_FATAL(cores_.num_cores() > 0, "CoreRangeSet must have at least one core");
    uint32_t num_cores = cores_.num_cores();
    auto shard_parameters = ShardSpecBuffer(cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
    ShardedBufferConfig sem_shard_config = {
        .device = device_,
        .size = num_cores * sizeof(uint32_t),
        .page_size = sizeof(uint32_t),
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_parameters),
    };
    buffer_ = distributed::AnyBuffer::create(sem_shard_config);

    this->reset_semaphore_value(initial_value);
}

IDevice* GlobalSemaphore::device() const { return device_; }

DeviceAddr GlobalSemaphore::address() const { return buffer_.get_buffer()->address(); }

void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const {
    // Write the initial value to the semaphore to the device
    // Only block for the slow dispatch case

    std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
    if (device_->using_slow_dispatch()) {
        detail::WriteToBuffer(*buffer_.get_buffer(), host_buffer);
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device_->id());
    } else {
        if (auto mesh_buffer = buffer_.get_mesh_buffer()) {
            distributed::EnqueueWriteMeshBuffer(mesh_buffer->device()->mesh_command_queue(), mesh_buffer, host_buffer);
        } else {
            EnqueueWriteBuffer(device_->command_queue(), *buffer_.get_buffer(), host_buffer, false);
        }
    }
}

}  // namespace tt::tt_metal

namespace std {

std::size_t hash<tt::tt_metal::GlobalSemaphore>::operator()(
    const tt::tt_metal::GlobalSemaphore& global_semaphore) const {
    return tt::stl::hash::hash_objects_with_default_seed(global_semaphore.attribute_values());
}

}  // namespace std
