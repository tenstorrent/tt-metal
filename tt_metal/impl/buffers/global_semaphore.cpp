// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
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

// GlobalSemaphore is implemented as a wrapper around a sharded buffer
// This can be updated in the future to be its own container with optimized dispatch functions
class GlobalSemaphore::Impl {
public:
    Impl(IDevice* device, const CoreRangeSet& cores, std::optional<uint32_t> initial_value, BufferType buffer_type,
         std::optional<uint64_t> address) :
        device_(device), cores_(cores) {
        this->setup_buffer(initial_value, buffer_type, address);
    }

    Impl(IDevice* device, CoreRangeSet&& cores, std::optional<uint32_t> initial_value, BufferType buffer_type,
         std::optional<uint64_t> address) :
        device_(device), cores_(std::move(cores)) {
        this->setup_buffer(initial_value, buffer_type, address);
    }

    IDevice* device() const { return device_; }

    const CoreRangeSet& cores() const { return cores_; }

    BufferType buffer_type() const { return buffer_.get_buffer()->buffer_type(); }

    DeviceAddr address() const { return buffer_.get_buffer()->address(); }

    void reset_semaphore_value(uint32_t reset_value) const {
        // Blocking write here to ensure that Global Semaphore reset value lands on
        // each physical device before the next program runs.
        // This is to ensure that cross-chip writes to the Global Semaphore are not
        // lost due to device skew.
        std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
        auto mesh_buffer = buffer_.get_mesh_buffer();
        bool using_fast_dispatch = MetalContext::instance().rtoptions().get_fast_dispatch();
        bool using_simulator = MetalContext::instance().rtoptions().get_simulator_enabled();
        if (using_fast_dispatch && !using_simulator) {
            distributed::EnqueueWriteMeshBuffer(
                mesh_buffer->device()->mesh_command_queue(), mesh_buffer, host_buffer, true);
        } else {
            auto* mesh_device = mesh_buffer->device();
            for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
                if (!mesh_device->is_local(coord)) {
                    continue;
                }
                tt::tt_metal::detail::WriteToBuffer(*mesh_buffer->get_device_buffer(coord), host_buffer);
            }
        }
    }

private:
    void setup_buffer(
        std::optional<uint32_t> initial_value, BufferType buffer_type, std::optional<uint64_t> address) {
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
        buffer_ = distributed::AnyBuffer::create(sem_shard_config, address);

        if (initial_value.has_value()) {
            this->reset_semaphore_value(initial_value.value());
        }
    }

    distributed::AnyBuffer buffer_;
    IDevice* device_;
    CoreRangeSet cores_;
};

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) :
    impl_(std::make_shared<Impl>(device, cores, initial_value, buffer_type, std::nullopt)) {}

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type) :
    impl_(std::make_shared<Impl>(device, std::move(cores), initial_value, buffer_type, std::nullopt)) {}

GlobalSemaphore::GlobalSemaphore(
    IDevice* device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address) :
    impl_(std::make_shared<Impl>(device, cores, initial_value, buffer_type, address)) {}

GlobalSemaphore::~GlobalSemaphore() = default;

IDevice* GlobalSemaphore::device() const { return impl_->device(); }

std::ostream& operator<<(std::ostream& os, const GlobalSemaphore& global_semaphore) {
    ttsl::reflection::operator<<(os, global_semaphore);
    return os;
}

DeviceAddr GlobalSemaphore::address() const { return impl_->address(); }

void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const { impl_->reset_semaphore_value(reset_value); }

std::tuple<CoreRangeSet, BufferType> GlobalSemaphore::attribute_values() const {
    return std::make_tuple(impl_->cores(), impl_->buffer_type());
}

}  // namespace tt::tt_metal

namespace std {

std::size_t hash<tt::tt_metal::GlobalSemaphore>::operator()(
    const tt::tt_metal::GlobalSemaphore& global_semaphore) const {
    return ttsl::hash::hash_objects_with_default_seed(global_semaphore.attribute_values());
}

}  // namespace std
