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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "global_semaphore_impl.hpp"
#include "global_semaphore_pool.hpp"
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
    device_{device.shared_from_this()}, cores_{std::move(cores)} {
    this->setup_buffer(device, initial_value, buffer_type, std::nullopt);
}

GlobalSemaphoreImpl::GlobalSemaphoreImpl(
    distributed::MeshDevice& device,
    CoreRangeSet cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address) :
    device_{device.shared_from_this()}, cores_{std::move(cores)} {
    this->setup_buffer(device, initial_value, buffer_type, address);
}

GlobalSemaphoreImpl::GlobalSemaphoreImpl(distributed::MeshDevice& device, CoreRangeSet cores) :
    device_{device.shared_from_this()}, cores_{std::move(cores)} {
    TT_FATAL(device.is_initialized(), "CreateGlobalSemaphore called on a closed MeshDevice");
    const auto& pool = device.impl().global_semaphore_pool();
    TT_FATAL(
        pool != nullptr,
        "This MeshDevice has no GlobalSemaphore pool. Call experimental::ReserveGlobalSemaphorePool right after "
        "opening it.");
    lease_ = pool->acquire(device, cores_);
    Inspector::global_semaphore_created(lease_.get(), lease_->address, cores_, device);
}

std::shared_ptr<distributed::MeshDevice> GlobalSemaphoreImpl::lock_device() const {
    auto device = device_.lock();
    TT_FATAL(device != nullptr, "GlobalSemaphore used after its MeshDevice was destroyed");
    return device;
}

distributed::MeshDevice& GlobalSemaphoreImpl::device() const { return *lock_device(); }

const CoreRangeSet& GlobalSemaphoreImpl::cores() const { return cores_; }

BufferType GlobalSemaphoreImpl::buffer_type() const {
    return lease_ ? BufferType::L1 : buffer_->device_local_config().buffer_type;
}

DeviceAddr GlobalSemaphoreImpl::address() const { return lease_ ? lease_->address : buffer_->address(); }

void GlobalSemaphoreImpl::reset_semaphore_value(
    uint32_t reset_value, ttsl::Span<const SubDeviceId> sub_device_ids) const {
    const auto mesh_device = lock_device();
    TT_FATAL(mesh_device->is_initialized(), "GlobalSemaphore reset after its MeshDevice was closed");
    if (lease_) {
        write_global_semaphore_value(*mesh_device, cores_, lease_->address, reset_value, sub_device_ids);
        Inspector::global_semaphore_reset(lease_.get(), reset_value);
        return;
    }
    TT_FATAL(sub_device_ids.empty(), "sub_device_ids are only supported for pool-backed GlobalSemaphores");
    // Blocking write here to ensure that Global Semaphore reset value lands on
    // each physical device before the next program runs.
    // This is to ensure that cross-chip writes to the Global Semaphore are not
    // lost due to device skew.
    std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
    const auto& rtoptions = mesh_device->impl().metal_env().get_rtoptions();
    bool using_fast_dispatch = rtoptions.get_fast_dispatch();
    bool using_simulator = rtoptions.get_simulator_enabled();
    if (using_fast_dispatch && !using_simulator) {
        mesh_device->mesh_command_queue().enqueue_write_mesh_buffer(buffer_, host_buffer.data(), /*blocking=*/true);
    } else {
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
            if (!mesh_device->is_local(coord)) {
                continue;
            }
            tt::tt_metal::slow_dispatch::WriteToBuffer(*buffer_->get_device_buffer(coord), host_buffer);
        }
    }
    Inspector::global_semaphore_reset(buffer_.get(), reset_value);
}

void GlobalSemaphoreImpl::setup_buffer(
    distributed::MeshDevice& device,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    std::optional<uint64_t> address) {
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
        &device,
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

void ReserveGlobalSemaphorePool(distributed::MeshDevice& device, uint32_t num_slots) {
    device.impl().reserve_global_semaphore_pool(num_slots);
}
}  // namespace experimental

// GlobalSemaphorePool implementation

GlobalSemaphorePoolReservation::GlobalSemaphorePoolReservation(
    AllocatorImpl& allocator, DeviceAddr size, DeviceAddr alignment) :
    allocator_{allocator} {
    const CoreRangeSet& worker_grid = allocator.get_config().worker_grid;
    const auto allocation = allocator.persistent_l1().allocate(worker_grid, size, alignment);
    allocation_id_ = allocation.id;
    address_ = allocation.address;

    const DeviceAddr end = allocation.address + allocation.size;
    for (const CoreCoord& core : corerange_to_cores(worker_grid)) {
        const auto* bank_ids = allocator.find_bank_ids(BufferType::L1, core);
        if (bank_ids == nullptr) {
            continue;
        }
        for (const uint32_t bank_id : *bank_ids) {
            const auto lowest_l1_buffer = allocator.get_lowest_occupied_l1_address(bank_id);
            if (lowest_l1_buffer.has_value() && *lowest_l1_buffer < end) {
                allocator.persistent_l1().deallocate(allocation_id_);
                TT_THROW(
                    "GlobalSemaphore pool [{}, {}) overlaps an L1 buffer at {} on core {}",
                    allocation.address,
                    end,
                    *lowest_l1_buffer,
                    core.str());
            }
        }
    }
}

GlobalSemaphorePoolReservation::~GlobalSemaphorePoolReservation() {
    try {
        allocator_.persistent_l1().deallocate(allocation_id_);
    } catch (...) {
        log_warning(tt::LogMetal, "GlobalSemaphore pool: persistent L1 release failed");
    }
}

GlobalSemaphorePool::GlobalSemaphorePool(
    std::unique_ptr<GlobalSemaphorePoolReservation> reservation,
    CoreRangeSet worker_grid,
    uint32_t num_slots,
    DeviceAddr slot_size) :
    reservation_{std::move(reservation)},
    base_address_{reservation_->address()},
    worker_grid_{std::move(worker_grid)},
    num_slots_{num_slots},
    slot_size_{slot_size},
    slot_in_use_(num_slots, false) {
    TT_FATAL(num_slots_ > 0, "GlobalSemaphore pool needs at least one slot");
}

void GlobalSemaphorePool::close() {
    std::lock_guard lock(mutex_);
    reservation_.reset();
    pending_.clear();
}

uint32_t GlobalSemaphorePool::num_live() const {
    std::lock_guard lock(mutex_);
    return num_live_;
}

uint32_t GlobalSemaphorePool::num_pending() const {
    std::lock_guard lock(mutex_);
    return static_cast<uint32_t>(pending_.size());
}

uint32_t GlobalSemaphorePool::peak_live() const {
    std::lock_guard lock(mutex_);
    return peak_live_;
}

void GlobalSemaphorePool::record_events_for_pending() {
    std::vector<std::shared_ptr<distributed::MeshDevice>> devices;
    {
        std::lock_guard lock(mutex_);
        for (const auto& pending : pending_) {
            if (pending.events.has_value()) {
                continue;
            }
            auto device = pending.device.lock();
            if (device != nullptr && std::find(devices.begin(), devices.end(), device) == devices.end()) {
                devices.push_back(std::move(device));
            }
        }
    }

    // Recorded without holding mutex_: command queues take the device's API lock.
    std::vector<std::pair<distributed::MeshDevice*, std::vector<distributed::MeshEvent>>> recorded;
    for (const auto& device : devices) {
        std::vector<distributed::MeshEvent> events;
        if (device->is_initialized() && device->impl().metal_env().get_rtoptions().get_fast_dispatch()) {
            for (uint8_t cq_id = 0; cq_id < device->num_hw_cqs(); cq_id++) {
                auto& cq = device->mesh_command_queue(cq_id);
                if (cq.trace_id().has_value()) {
                    continue;  // A capturing queue runs nothing until the trace is replayed.
                }
                events.push_back(cq.enqueue_record_event_to_host());
            }
        }
        recorded.emplace_back(device.get(), std::move(events));
    }

    std::lock_guard lock(mutex_);
    for (auto& pending : pending_) {
        if (pending.events.has_value()) {
            continue;
        }
        const auto device = pending.device.lock();
        const auto it = std::find_if(
            recorded.begin(), recorded.end(), [&](const auto& entry) { return entry.first == device.get(); });
        if (device == nullptr) {
            pending.events.emplace();
        } else if (it != recorded.end()) {
            pending.events = it->second;
        }
    }
}

void GlobalSemaphorePool::collect_completed_pending() {
    std::erase_if(pending_, [this](const PendingSlot& pending) {
        if (!pending.events.has_value()) {
            return false;
        }
        const auto device = pending.device.lock();
        const bool done = device == nullptr || !device->is_initialized() ||
                          std::all_of(pending.events->begin(), pending.events->end(), [](const auto& event) {
                              return distributed::EventQuery(event);
                          });
        if (done) {
            slot_in_use_[pending.slot] = false;
        }
        return done;
    });
}

void GlobalSemaphorePool::device_quiesced(const distributed::MeshDeviceImpl& device) {
    std::lock_guard lock(mutex_);
    std::erase_if(pending_, [&](const PendingSlot& pending) {
        const auto pending_device = pending.device.lock();
        const bool done = pending.events.has_value() && pending_device != nullptr && &pending_device->impl() == &device;
        if (done) {
            slot_in_use_[pending.slot] = false;
        }
        return done;
    });
}

std::shared_ptr<const GlobalSemaphoreSlotLease> GlobalSemaphorePool::acquire(
    distributed::MeshDevice& device, const CoreRangeSet& cores) {
    TT_FATAL(cores.num_cores() > 0, "GlobalSemaphore needs at least one core");
    TT_FATAL(
        cores.intersection(worker_grid_).num_cores() == cores.num_cores(),
        "GlobalSemaphore cores {} are not all worker cores of the device ({})",
        cores.str(),
        worker_grid_.str());

    uint32_t slot = num_slots_;
    std::optional<PendingSlot> wait_for;
    while (slot == num_slots_) {
        record_events_for_pending();
        std::lock_guard lock(mutex_);
        TT_FATAL(reservation_ != nullptr, "GlobalSemaphore created after its MeshDevice was closed");
        collect_completed_pending();
        const auto free_slot = std::find(slot_in_use_.begin(), slot_in_use_.end(), false);
        if (free_slot != slot_in_use_.end()) {
            slot = static_cast<uint32_t>(free_slot - slot_in_use_.begin());
            slot_in_use_[slot] = true;
            break;
        }
        TT_FATAL(
            !pending_.empty(),
            "GlobalSemaphore pool is full: all {} slots are in use (peak {}). Reserve a larger pool with "
            "experimental::ReserveGlobalSemaphorePool, or set TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS.",
            num_slots_,
            peak_live_);
        // Wait for the oldest returned slot that has its events recorded. Slots returned since
        // record_events_for_pending() have none yet; if those are all that's left, loop and record them.
        const auto tagged = std::find_if(
            pending_.begin(), pending_.end(), [](const auto& pending) { return pending.events.has_value(); });
        if (tagged != pending_.end()) {
            wait_for = std::move(*tagged);
            pending_.erase(tagged);
            slot = wait_for->slot;
        }
    }
    {
        std::lock_guard lock(mutex_);
        ++num_live_;
        peak_live_ = std::max(peak_live_, num_live_);
    }

    std::shared_ptr<const GlobalSemaphoreSlotLease> lease;
    try {
        lease = std::make_shared<const GlobalSemaphoreSlotLease>(
            weak_from_this(), device.shared_from_this(), cores, base_address() + (slot * slot_size_), slot);
    } catch (...) {
        std::lock_guard lock(mutex_);
        if (wait_for.has_value()) {
            pending_.push_back(std::move(*wait_for));
        } else {
            slot_in_use_[slot] = false;
        }
        --num_live_;
        throw;
    }

    // From here the lease owns the slot, so a failure below hands it back.
    if (wait_for.has_value()) {
        const auto previous_device = wait_for->device.lock();
        if (previous_device != nullptr && previous_device->is_initialized()) {
            for (const auto& event : *wait_for->events) {
                distributed::EventSynchronize(event);
            }
        }
    }
    write_global_semaphore_value(device, cores, lease->address, 0);
    return lease;
}

void GlobalSemaphorePool::release(uint32_t slot, std::weak_ptr<distributed::MeshDevice> device) noexcept {
    try {
        std::lock_guard lock(mutex_);
        pending_.push_back(PendingSlot{.slot = slot, .device = std::move(device), .events = std::nullopt});
        --num_live_;
    } catch (...) {
        log_warning(tt::LogMetal, "GlobalSemaphore pool: failed to return slot {}", slot);
    }
}

GlobalSemaphoreSlotLease::GlobalSemaphoreSlotLease(
    std::weak_ptr<GlobalSemaphorePool> pool,
    std::weak_ptr<distributed::MeshDevice> device,
    CoreRangeSet cores,
    DeviceAddr address,
    uint32_t slot) :
    pool{std::move(pool)}, device{std::move(device)}, cores{std::move(cores)}, address{address}, slot{slot} {}

GlobalSemaphoreSlotLease::~GlobalSemaphoreSlotLease() {
    Inspector::global_semaphore_destroyed(this);
    if (auto locked = pool.lock()) {
        locked->release(slot, device);
    }
}

void write_global_semaphore_value(
    distributed::MeshDevice& device,
    const CoreRangeSet& cores,
    DeviceAddr address,
    uint32_t value,
    ttsl::Span<const SubDeviceId> sub_device_ids) {
    const auto& rtoptions = device.impl().metal_env().get_rtoptions();
    if (rtoptions.get_fast_dispatch() && !rtoptions.get_simulator_enabled()) {
        // Blocking, so the value lands on every chip before the next program runs. Otherwise device skew
        // could let a reset overwrite an increment sent from another chip.
        auto& mesh_cq = dynamic_cast<distributed::FDMeshCommandQueue&>(device.mesh_command_queue());
        mesh_cq.enqueue_write_to_core_ranges(
            cores, address, std::as_bytes(std::span<const uint32_t>(&value, 1)), /*blocking=*/true, sub_device_ids);
        return;
    }

    const auto& cluster = device.impl().metal_env().get_cluster();
    // ranges() returns a view, so the merged set must stay alive for the loop.
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

void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value, ttsl::Span<const SubDeviceId> sub_device_ids) const {
    impl().reset_semaphore_value(reset_value, sub_device_ids);
}

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
