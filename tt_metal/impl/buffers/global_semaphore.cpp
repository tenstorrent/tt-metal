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
#include <tt-metalium/tt_align.hpp>
#include <tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/global_semaphore_space.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "global_semaphore_impl.hpp"
#include "global_semaphore_space_impl.hpp"
#include "distributed/fd_mesh_command_queue.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/context/context_types.hpp"
#include "impl/dispatch/host_device_transfer.hpp"
#include "llrt/tt_cluster.hpp"
#include "mesh_device.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_env_impl.hpp"
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

GlobalSemaphoreImpl::GlobalSemaphoreImpl(std::shared_ptr<const experimental::GlobalSemaphoreLease> lease) :
    lease_{std::move(lease)}, device_{&lease_->space->device()}, cores_{lease_->cores} {}

distributed::MeshDevice& GlobalSemaphoreImpl::device() const { return *device_; }

const CoreRangeSet& GlobalSemaphoreImpl::cores() const { return cores_; }

BufferType GlobalSemaphoreImpl::buffer_type() const {
    return lease_ ? BufferType::L1 : buffer_->device_local_config().buffer_type;
}

DeviceAddr GlobalSemaphoreImpl::address() const { return lease_ ? lease_->address : buffer_->address(); }

void GlobalSemaphoreImpl::reset_semaphore_value(uint32_t reset_value) const {
    if (lease_) {
        experimental::write_global_semaphore_value(*device_, cores_, lease_->address, reset_value);
        return;
    }
    // Blocking write here to ensure that Global Semaphore reset value lands on
    // each physical device before the next program runs.
    // This is to ensure that cross-chip writes to the Global Semaphore are not
    // lost due to device skew.
    std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
    auto& mesh_device = device();
    const auto& rtoptions = mesh_device.impl().metal_env().get_rtoptions();
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
    // Validate every destination in its selected SRAM region before any constructor can write.
    for (auto* local_device : device.get_devices()) {
        const auto& allocator = local_device->allocator_impl();
        const auto footprint = tt::align(sizeof(uint32_t), allocator->get_alignment(BufferType::L1));
        allocator->validate_l1_range(address, footprint, buffer_type);
    }
    return GlobalSemaphore{GlobalSemaphoreImpl{device, cores, initial_value, buffer_type, address}};
}

// GlobalSemaphoreSpace implementation

namespace {

bool is_subset(const CoreRangeSet& subset, const CoreRangeSet& superset) {
    return subset.intersection(superset).num_cores() == subset.num_cores();
}

}  // namespace

GlobalSemaphoreSpaceImpl::GlobalSemaphoreSpaceImpl(
    distributed::MeshDevice& device, CoreRangeSet cores, uint32_t capacity) :
    device_{&device}, cores_{std::move(cores)}, capacity_{capacity}, slot_in_use_(capacity, false) {
    TT_FATAL(cores_.num_cores() > 0, "GlobalSemaphoreSpace needs at least one core");
    TT_FATAL(capacity_ > 0, "GlobalSemaphoreSpace capacity must be non-zero");

    const auto& hal = MetalContext::instance(extract_context_id(device_)).hal();
    auto& allocator = *device_->allocator_impl();
    slot_stride_ = hal.get_alignment(HalMemType::L1);

    const auto allocation =
        allocator.persistent_l1().allocate(cores_, static_cast<DeviceAddr>(capacity_) * slot_stride_, slot_stride_);
    allocation_id_ = allocation.id;
    base_address_ = allocation.address;

    try {
        const DeviceAddr end = allocation.address + allocation.size;
        for (const CoreCoord& core : corerange_to_cores(cores_)) {
            const auto& bank_ids = allocator.get_bank_ids_from_logical_core(BufferType::L1, core);
            TT_FATAL(bank_ids.size() == 1, "Expected one L1 bank for GlobalSemaphoreSpace core {}", core.str());
            const auto lowest_l1_buffer = allocator.get_lowest_occupied_l1_address(bank_ids.front());
            TT_FATAL(
                !lowest_l1_buffer.has_value() || end <= *lowest_l1_buffer,
                "GlobalSemaphoreSpace region [{}, {}) overlaps an L1 buffer at {} on core {}",
                allocation.address,
                end,
                lowest_l1_buffer.value_or(0),
                core.str());
        }
    } catch (...) {
        allocator.persistent_l1().deallocate(allocation_id_);
        allocation_id_ = 0;
        throw;
    }
}

GlobalSemaphoreSpaceImpl::~GlobalSemaphoreSpaceImpl() {
    if (allocation_id_ == 0 || !device_->is_initialized()) {
        return;
    }
    try {
        device_->allocator_impl()->persistent_l1().deallocate(allocation_id_);
    } catch (...) {
        // Destructors must not throw. A missing allocation indicates an internal lifetime error.
        log_warning(tt::LogMetal, "GlobalSemaphoreSpace: persistent L1 release failed");
    }
}

uint32_t GlobalSemaphoreSpaceImpl::num_live_semaphores() const {
    std::lock_guard lock(mutex_);
    return num_live_;
}

std::shared_ptr<const GlobalSemaphoreLease> GlobalSemaphoreSpaceImpl::acquire(const CoreRangeSet& cores) {
    TT_FATAL(cores.num_cores() > 0, "GlobalSemaphore needs at least one core");
    TT_FATAL(
        is_subset(cores, cores_),
        "GlobalSemaphore cores {} are not all in the GlobalSemaphoreSpace cores {}",
        cores.str(),
        cores_.str());

    uint32_t slot = 0;
    {
        std::lock_guard lock(mutex_);
        while (slot < capacity_ && slot_in_use_[slot]) {
            ++slot;
        }
        TT_FATAL(
            slot < capacity_,
            "GlobalSemaphoreSpace is full: all {} semaphores are live. Create the space with a larger capacity.",
            capacity_);
        slot_in_use_[slot] = true;
        ++num_live_;
    }

    // The lease owns the slot from here on, so a failed reset below returns it.
    auto lease = std::make_shared<const GlobalSemaphoreLease>(
        shared_from_this(), cores, base_address_ + (slot * slot_stride_), slot);
    write_global_semaphore_value(*device_, cores, lease->address, 0);
    return lease;
}

void GlobalSemaphoreSpaceImpl::release(uint32_t slot) noexcept {
    std::lock_guard lock(mutex_);
    slot_in_use_[slot] = false;
    --num_live_;
}

GlobalSemaphoreLease::GlobalSemaphoreLease(
    std::shared_ptr<GlobalSemaphoreSpaceImpl> space, CoreRangeSet cores, DeviceAddr address, uint32_t slot) :
    space{std::move(space)}, cores{std::move(cores)}, address{address}, slot{slot} {}

GlobalSemaphoreLease::~GlobalSemaphoreLease() { space->release(slot); }

void write_global_semaphore_value(
    distributed::MeshDevice& device, const CoreRangeSet& cores, DeviceAddr address, uint32_t value) {
    const auto& rtoptions = MetalContext::instance(extract_context_id(&device)).rtoptions();
    if (rtoptions.get_fast_dispatch() && !rtoptions.get_simulator_enabled()) {
        // Blocking, so the value has landed on every device before the next program runs and a
        // cross-chip increment cannot be overwritten because of device skew.
        auto& mesh_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(device.mesh_command_queue());
        mesh_cq.enqueue_write_to_core_ranges(
            cores, address, std::as_bytes(std::span<const uint32_t>(&value, 1)), /*blocking=*/true);
        return;
    }

    const auto& cluster = MetalContext::instance(extract_context_id(&device)).get_cluster();
    // Named: ranges() returns a view into the merged set, which must outlive the loop.
    const CoreRangeSet merged = cores.merge_ranges();
    for (IDevice* target_device : device.get_devices()) {
        for (const CoreRange& range : merged.ranges()) {
            if (range.size() == 1) {
                std::vector<uint32_t> words{value};
                TT_FATAL(
                    tt::tt_metal::detail::WriteToDeviceL1(target_device, range.start_coord, address, words),
                    "Failed to write GlobalSemaphore on core {} of device {}",
                    range.start_coord.str(),
                    target_device->id());
                continue;
            }
            const CoreCoord start = target_device->virtual_core_from_logical_core(range.start_coord, CoreType::WORKER);
            const CoreCoord end = target_device->virtual_core_from_logical_core(range.end_coord, CoreType::WORKER);
            cluster.noc_multicast_write(&value, sizeof(value), target_device->id(), start, end, address);
        }
        cluster.l1_barrier(target_device->id());
    }
}

GlobalSemaphoreSpace::GlobalSemaphoreSpace(std::shared_ptr<GlobalSemaphoreSpaceImpl> impl) : impl_{std::move(impl)} {}
GlobalSemaphoreSpace::GlobalSemaphoreSpace(GlobalSemaphoreSpace&&) noexcept = default;
GlobalSemaphoreSpace& GlobalSemaphoreSpace::operator=(GlobalSemaphoreSpace&&) noexcept = default;
GlobalSemaphoreSpace::~GlobalSemaphoreSpace() = default;

const CoreRangeSet& GlobalSemaphoreSpace::cores() const { return impl_->cores(); }

uint32_t GlobalSemaphoreSpace::capacity() const { return impl_->capacity(); }

uint32_t GlobalSemaphoreSpace::num_live_semaphores() const { return impl_->num_live_semaphores(); }

GlobalSemaphore GlobalSemaphoreSpace::create_semaphore(const CoreRangeSet& cores) {
    return GlobalSemaphore{GlobalSemaphoreImpl{impl_->acquire(cores)}};
}

GlobalSemaphoreSpace CreateGlobalSemaphoreSpace(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t capacity) {
    return GlobalSemaphoreSpace(std::make_shared<GlobalSemaphoreSpaceImpl>(device, cores, capacity));
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

std::tuple<CoreRangeSet, BufferType> GlobalSemaphore::attribute_values() const {
    return std::make_tuple(impl().cores(), impl().buffer_type());
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
