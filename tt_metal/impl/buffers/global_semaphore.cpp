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

#include "global_semaphore_impl.hpp"
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// GlobalSemaphoreImpl implementation

GlobalSemaphoreImpl::GlobalSemaphoreImpl(
    IDevice* device, const CoreRangeSet& cores, std::optional<uint32_t> initial_value, BufferType buffer_type) :
    device_(device), cores_(cores) {
    this->setup_buffer(initial_value, buffer_type, std::nullopt);
}

GlobalSemaphoreImpl::GlobalSemaphoreImpl(
    IDevice* device, CoreRangeSet&& cores, std::optional<uint32_t> initial_value, BufferType buffer_type) :
    device_(device), cores_(std::move(cores)) {
    this->setup_buffer(initial_value, buffer_type, std::nullopt);
}

GlobalSemaphoreImpl::GlobalSemaphoreImpl(
    IDevice* device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address) :
    device_(device), cores_(cores) {
    this->setup_buffer(initial_value, buffer_type, address);
}

IDevice* GlobalSemaphoreImpl::device() const { return device_; }

const CoreRangeSet& GlobalSemaphoreImpl::cores() const { return cores_; }

BufferType GlobalSemaphoreImpl::buffer_type() const { return buffer_.get_buffer()->buffer_type(); }

DeviceAddr GlobalSemaphoreImpl::address() const { return buffer_.get_buffer()->address(); }

void GlobalSemaphoreImpl::reset_semaphore_value(uint32_t reset_value) const {
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

void GlobalSemaphoreImpl::setup_buffer(
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

namespace experimental {
// Forge backdoor API.
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address) {
    return GlobalSemaphore(GlobalSemaphoreImpl(device, cores, initial_value, buffer_type, address));
}
}  // namespace experimental

// GlobalSemaphore implementation

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) :
    GlobalSemaphore(GlobalSemaphoreImpl(device, cores, initial_value, buffer_type)) {}

GlobalSemaphore::GlobalSemaphore(
    IDevice* device, CoreRangeSet&& cores, uint32_t initial_value, BufferType buffer_type) :
    GlobalSemaphore(GlobalSemaphoreImpl(device, std::move(cores), initial_value, buffer_type)) {}

GlobalSemaphore::GlobalSemaphore(GlobalSemaphoreImpl&& impl) :
    pimpl_(std::make_unique<GlobalSemaphoreImpl>(std::move(impl))) {}

GlobalSemaphore::GlobalSemaphore(const GlobalSemaphore& other) :
    pimpl_(other.pimpl_ ? std::make_unique<GlobalSemaphoreImpl>(*other.pimpl_) : nullptr) {}

GlobalSemaphore& GlobalSemaphore::operator=(const GlobalSemaphore& other) {
    if (this != &other) {
        pimpl_ = other.pimpl_ ? std::make_unique<GlobalSemaphoreImpl>(*other.pimpl_) : nullptr;
    }
    return *this;
}

GlobalSemaphore::GlobalSemaphore(GlobalSemaphore&& other) noexcept = default;

GlobalSemaphore& GlobalSemaphore::operator=(GlobalSemaphore&& other) noexcept = default;

GlobalSemaphore::~GlobalSemaphore() = default;

IDevice* GlobalSemaphore::device() const { return pimpl_->device(); }

std::ostream& operator<<(std::ostream& os, const GlobalSemaphore& global_semaphore) {
    ttsl::reflection::operator<<(os, global_semaphore);
    return os;
}

DeviceAddr GlobalSemaphore::address() const { return pimpl_->address(); }

void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const { pimpl_->reset_semaphore_value(reset_value); }

std::tuple<CoreRangeSet, BufferType> GlobalSemaphore::attribute_values() const {
    return std::make_tuple(pimpl_->cores(), pimpl_->buffer_type());
}

std::vector<uint32_t> GlobalSemaphore::read_semaphore_values() const {
    // Blocking read of the sem buffer; one uint32 per core in cores_, per device,
    // concatenated in MeshCoordinate order. The semaphore buffer is replicated (not
    // sharded) across the mesh, so we read each device's shard individually:
    // EnqueueReadMeshBuffer only supports SHARDED layouts (or a unit mesh), and would
    // otherwise TT_FATAL on a multi-device replicated buffer.
    std::vector<uint32_t> host_buffer;
    auto mesh_buffer = buffer_.get_mesh_buffer();
    auto* mesh_device = mesh_buffer->device();
    host_buffer.reserve(cores_.num_cores() * mesh_device->shape().mesh_size());
    bool using_fast_dispatch = MetalContext::instance().rtoptions().get_fast_dispatch();
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        std::vector<uint32_t> shard_buffer;
        if (using_fast_dispatch) {
            distributed::ReadShard(
                mesh_device->mesh_command_queue(), shard_buffer, mesh_buffer, coord, /*blocking=*/true);
        } else {
            tt::tt_metal::detail::ReadFromBuffer(*mesh_buffer->get_device_buffer(coord), shard_buffer);
        }
        host_buffer.insert(host_buffer.end(), shard_buffer.begin(), shard_buffer.end());
    }
    return host_buffer;
}

}  // namespace tt::tt_metal

namespace std {

std::size_t hash<tt::tt_metal::GlobalSemaphore>::operator()(
    const tt::tt_metal::GlobalSemaphore& global_semaphore) const {
    return ttsl::hash::hash_objects_with_default_seed(global_semaphore.attribute_values());
}

}  // namespace std
