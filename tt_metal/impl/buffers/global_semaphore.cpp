// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include <global_semaphore.hpp>
#include <tt-metalium/experimental/global_semaphore.hpp>
#include <host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt_metal.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "global_semaphore_impl.hpp"
#include "distributed/fd_mesh_command_queue.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/dispatch/host_device_transfer.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-logger/tt-logger.hpp>
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/debug/inspector/inspector.hpp"

namespace tt::tt_metal {

// GlobalSemaphoreImpl implementation

GlobalSemaphoreImpl::GlobalSemaphoreImpl(
    distributed::MeshDevice& device,
    CoreRangeSet cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type) :
    device_{&device}, cores_{std::move(cores)} {
    this->setup_buffer(initial_value, buffer_type, std::nullopt);
}

GlobalSemaphoreImpl::GlobalSemaphoreImpl(
    distributed::MeshDevice& device,
    CoreRangeSet cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address) :
    device_{&device}, cores_{std::move(cores)} {
    this->setup_buffer(initial_value, buffer_type, address);
}

GlobalSemaphoreImpl::GlobalSemaphoreImpl(
    distributed::MeshDevice& device,
    CoreRangeSet cores,
    std::optional<uint32_t> initial_value,
    PersistentL1Placement /*placement*/) :
    device_{&device}, cores_{std::move(cores)} {
    this->setup_persistent(initial_value);
}

GlobalSemaphoreImpl::PersistentAllocation::~PersistentAllocation() {
    if (device == nullptr || id == 0 || !device->is_initialized()) {
        return;
    }
    try {
        device->allocator_impl()->persistent_l1().deallocate(id);
    } catch (...) {
        // Destructors must not throw. A missing allocation indicates an internal lifetime error.
        log_warning(tt::LogMetal, "GlobalSemaphore: persistent L1 release failed with unknown exception");
    }
}

distributed::MeshDevice& GlobalSemaphoreImpl::device() const { return *device_; }

const CoreRangeSet& GlobalSemaphoreImpl::cores() const { return cores_; }

BufferType GlobalSemaphoreImpl::buffer_type() const {
    return persistent_ ? BufferType::L1 : buffer_->device_local_config().buffer_type;
}

DeviceAddr GlobalSemaphoreImpl::address() const { return persistent_ ? persistent_->address : buffer_->address(); }

bool GlobalSemaphoreImpl::is_persistent() const { return persistent_ != nullptr; }

void GlobalSemaphoreImpl::setup_persistent(std::optional<uint32_t> initial_value) {
    TT_FATAL(cores_.num_cores() > 0, "CoreRangeSet must have at least one core");
    const auto& hal = MetalContext::instance(extract_context_id(device_)).hal();
    auto& allocator = *device_->allocator_impl();

    auto allocation = allocator.persistent_l1().allocate(cores_, sizeof(uint32_t), hal.get_alignment(HalMemType::L1));
    persistent_ = std::make_shared<PersistentAllocation>();
    persistent_->device = device_;
    persistent_->id = allocation.id;
    persistent_->address = allocation.address;

    const DeviceAddr end = allocation.address + allocation.size;
    for (const CoreCoord& core : corerange_to_cores(cores_)) {
        const auto& bank_ids = allocator.get_bank_ids_from_logical_core(BufferType::L1, core);
        TT_FATAL(bank_ids.size() == 1, "Expected one L1 bank for GlobalSemaphore core {}", core.str());
        const auto lowest_global_allocation = allocator.get_lowest_occupied_l1_address(bank_ids.front());
        TT_FATAL(
            !lowest_global_allocation.has_value() || end <= *lowest_global_allocation,
            "GlobalSemaphore persistent L1 slot [{}, {}) overlaps an existing L1 allocation at {} on core {}",
            allocation.address,
            end,
            lowest_global_allocation.value_or(0),
            core.str());
    }

    if (initial_value.has_value()) {
        this->reset_persistent_value(initial_value.value());
    }
}

void GlobalSemaphoreImpl::reset_persistent_value(uint32_t reset_value) const {
    const DeviceAddr address = persistent_->address;
    const auto& rtoptions = MetalContext::instance(extract_context_id(device_)).rtoptions();
    if (rtoptions.get_fast_dispatch() && !rtoptions.get_simulator_enabled()) {
        // Blocking for the same reason as the buffer-backed path below.
        auto& mesh_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(device_->mesh_command_queue());
        mesh_cq.enqueue_write_to_core_ranges(cores_, address, &reset_value, sizeof(reset_value), /*blocking=*/true);
        return;
    }

    const auto& cluster = MetalContext::instance(extract_context_id(device_)).get_cluster();
    // Named: ranges() returns a view into the merged set, which must outlive the loop.
    const CoreRangeSet merged = cores_.merge_ranges();
    for (IDevice* target_device : device_->get_devices()) {
        for (const CoreRange& range : merged.ranges()) {
            if (range.size() == 1) {
                std::vector<uint32_t> value{reset_value};
                TT_FATAL(
                    detail::WriteToDeviceL1(target_device, range.start_coord, address, value),
                    "Failed to reset GlobalSemaphore on core {} of device {}",
                    range.start_coord.str(),
                    target_device->id());
                continue;
            }
            const CoreCoord start = target_device->virtual_core_from_logical_core(range.start_coord, CoreType::WORKER);
            const CoreCoord end = target_device->virtual_core_from_logical_core(range.end_coord, CoreType::WORKER);
            cluster.noc_multicast_write(&reset_value, sizeof(reset_value), target_device->id(), start, end, address);
        }
        cluster.l1_barrier(target_device->id());
    }
}

void GlobalSemaphoreImpl::reset_semaphore_value(uint32_t reset_value) const {
    if (persistent_) {
        this->reset_persistent_value(reset_value);
        return;
    }
    // Blocking write here to ensure that Global Semaphore reset value lands on
    // each physical device before the next program runs.
    // This is to ensure that cross-chip writes to the Global Semaphore are not
    // lost due to device skew.
    std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
    auto& mesh_device = device();
    const auto& rtoptions = MetalContext::instance(extract_context_id(&mesh_device)).rtoptions();
    bool using_fast_dispatch = rtoptions.get_fast_dispatch();
    bool using_simulator = rtoptions.get_simulator_enabled();
    if (using_fast_dispatch && !using_simulator) {
        mesh_device.mesh_command_queue().enqueue_write_mesh_buffer(buffer_, host_buffer.data(), /*blocking=*/true);
    } else {
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
            if (!mesh_device.is_local(coord)) {
                continue;
            }
            tt::tt_metal::slow_dispatch::WriteToBuffer(*buffer_->get_device_buffer(coord), host_buffer);
        }
    }
    Inspector::global_semaphore_reset(buffer_.get(), reset_value);
}

void GlobalSemaphoreImpl::setup_buffer(
    std::optional<uint32_t> initial_value, BufferType buffer_type, std::optional<uint64_t> address) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global semaphore can only be created for L1 buffer types");
    TT_FATAL(cores_.num_cores() > 0, "CoreRangeSet must have at least one core");
    uint32_t num_cores = cores_.num_cores();
    auto shard_parameters = ShardSpecBuffer(cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
    buffer_ = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = num_cores * sizeof(uint32_t)},
        distributed::DeviceLocalBufferConfig{
            .page_size = sizeof(uint32_t),
            .buffer_type = buffer_type,
            .sharding_args = BufferShardingArgs(std::move(shard_parameters), TensorMemoryLayout::HEIGHT_SHARDED),
        },
        device_,
        address);
    Inspector::global_semaphore_created(buffer_.get(), cores_);

    if (initial_value.has_value()) {
        this->reset_semaphore_value(initial_value.value());
    }
}

namespace experimental {
// Forge backdoor API.
GlobalSemaphore CreateGlobalSemaphore(
    distributed::MeshDevice& device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address) {
    return GlobalSemaphore{GlobalSemaphoreImpl{device, cores, initial_value, buffer_type, address}};
}

GlobalSemaphore CreatePersistentGlobalSemaphore(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t initial_value) {
    return GlobalSemaphore{GlobalSemaphoreImpl{device, cores, initial_value, PersistentL1Placement{}}};
}
}  // namespace experimental

// GlobalSemaphore implementation

GlobalSemaphore::GlobalSemaphore(
    distributed::MeshDevice& device, CoreRangeSet cores, uint32_t initial_value, BufferType buffer_type) :
    GlobalSemaphore{GlobalSemaphoreImpl{device, std::move(cores), initial_value, buffer_type}} {}

GlobalSemaphore::GlobalSemaphore(GlobalSemaphoreImpl impl) :
    impl_(std::make_unique<GlobalSemaphoreImpl>(std::move(impl))) {}

GlobalSemaphore::GlobalSemaphore(const GlobalSemaphore& other) :
    impl_(other.impl_ ? std::make_unique<GlobalSemaphoreImpl>(*other.impl_) : nullptr) {}

GlobalSemaphore& GlobalSemaphore::operator=(const GlobalSemaphore& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<GlobalSemaphoreImpl>(*other.impl_) : nullptr;
    }
    return *this;
}

GlobalSemaphore::GlobalSemaphore(GlobalSemaphore&& other) noexcept = default;

GlobalSemaphore& GlobalSemaphore::operator=(GlobalSemaphore&& other) noexcept = default;

GlobalSemaphore::~GlobalSemaphore() = default;

IDevice* GlobalSemaphore::device() const { return &impl().device(); }

GlobalSemaphoreImpl& GlobalSemaphore::impl() {
    TT_FATAL(impl_ != nullptr, "GlobalSemaphore is in a moved-from state.");
    return *impl_;
}

const GlobalSemaphoreImpl& GlobalSemaphore::impl() const {
    TT_FATAL(impl_ != nullptr, "GlobalSemaphore is in a moved-from state.");
    return *impl_;
}

DeviceAddr GlobalSemaphore::address() const { return impl().address(); }

void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const { impl().reset_semaphore_value(reset_value); }

std::tuple<CoreRangeSet, BufferType, bool> GlobalSemaphore::attribute_values() const {
    return std::make_tuple(impl().cores(), impl().buffer_type(), impl().is_persistent());
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::GlobalSemaphore& global_semaphore) {
    ttsl::reflection::operator<<(os, global_semaphore);
    return os;
}

namespace std {

std::size_t hash<tt::tt_metal::GlobalSemaphore>::operator()(
    const tt::tt_metal::GlobalSemaphore& global_semaphore) const {
    return ttsl::hash::hash_objects_with_default_seed(global_semaphore.attribute_values());
}

}  // namespace std
