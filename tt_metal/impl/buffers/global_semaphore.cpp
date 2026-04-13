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
#include <tt-metalium/mesh_buffer.hpp>
#include <tt_metal.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

struct GlobalSemaphore::Impl {
    Impl(
        IDevice* device,
        const CoreRangeSet& cores,
        std::optional<uint32_t> initial_value,
        BufferType buffer_type,
        std::optional<uint64_t> address) :
        device_(device), cores_(cores) {
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
            reset(initial_value.value());
        }
    }

    void reset(uint32_t reset_value) const {
        std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
        auto mesh_buffer = buffer_.get_mesh_buffer();
        bool using_fast_dispatch = MetalContext::instance().rtoptions().get_fast_dispatch();
        if (using_fast_dispatch) {
            distributed::EnqueueWriteMeshBuffer(
                mesh_buffer->device()->mesh_command_queue(), mesh_buffer, host_buffer, true);
        } else {
            for (const auto& coord : distributed::MeshCoordinateRange(mesh_buffer->device()->shape())) {
                tt::tt_metal::detail::WriteToBuffer(*mesh_buffer->get_device_buffer(coord), host_buffer);
            }
        }
    }

    // GlobalSemaphore is implemented as a wrapper around a sharded buffer
    distributed::AnyBuffer buffer_;
    IDevice* device_;
    CoreRangeSet cores_;
};

// ---------------------------------------------------------------------------
// GlobalSemaphore
// ---------------------------------------------------------------------------

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) :
    pimpl_(std::make_unique<Impl>(device, cores, initial_value, buffer_type, std::nullopt)) {}

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type) :
    pimpl_(std::make_unique<Impl>(device, std::move(cores), initial_value, buffer_type, std::nullopt)) {}

GlobalSemaphore::GlobalSemaphore(
    IDevice* device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address) :
    pimpl_(std::make_unique<Impl>(device, cores, initial_value, buffer_type, address)) {}

GlobalSemaphore::GlobalSemaphore(const GlobalSemaphore& other) : pimpl_(std::make_unique<Impl>(*other.pimpl_)) {}

GlobalSemaphore& GlobalSemaphore::operator=(const GlobalSemaphore& other) {
    if (this != &other) {
        pimpl_ = std::make_unique<Impl>(*other.pimpl_);
    }
    return *this;
}

GlobalSemaphore::GlobalSemaphore(GlobalSemaphore&&) noexcept = default;
GlobalSemaphore& GlobalSemaphore::operator=(GlobalSemaphore&&) noexcept = default;
GlobalSemaphore::~GlobalSemaphore() = default;

IDevice* GlobalSemaphore::device() const { return pimpl_->device_; }

DeviceAddr GlobalSemaphore::address() const { return pimpl_->buffer_.get_buffer()->address(); }

std::tuple<CoreRangeSet, BufferType> GlobalSemaphore::attribute_values() const {
    return std::make_tuple(pimpl_->cores_, pimpl_->buffer_.get_buffer()->buffer_type());
}

void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const {
    // Blocking write here to ensure that Global Semaphore reset value lands on
    // each physical device before the next program runs.
    // This is to ensure that cross-chip writes to the Global Semaphore are not
    // lost due to device skew.
    pimpl_->reset(reset_value);
}

std::ostream& operator<<(std::ostream& os, const GlobalSemaphore& global_semaphore) {
    tt::stl::reflection::operator<<(os, global_semaphore);
    return os;
}

}  // namespace tt::tt_metal

namespace std {

std::size_t hash<tt::tt_metal::GlobalSemaphore>::operator()(
    const tt::tt_metal::GlobalSemaphore& global_semaphore) const {
    return tt::stl::hash::hash_objects_with_default_seed(global_semaphore.attribute_values());
}

}  // namespace std
