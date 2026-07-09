
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <mesh_buffer.hpp>
#include <mesh_coord.hpp>
#include <mesh_event.hpp>
#include <tt_stl/overloaded.hpp>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>
#include "device.hpp"
#include "impl/allocator/allocator.hpp"
#include "mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"

namespace per_core_allocation = tt::tt_metal::experimental::per_core_allocation;

namespace tt::tt_metal::distributed {
namespace {

void validate_mesh_buffer_config(const MeshBufferConfig& config, const MeshDevice& mesh_device) {
    if (std::holds_alternative<ReplicatedBufferConfig>(config)) {
        // Nothing to validate.
        return;
    }

    const auto& sharded_config = std::get<ShardedBufferConfig>(config);
    const auto [global_buffer_height, global_buffer_width] = sharded_config.global_buffer_shape;
    const auto [shard_height, shard_width] = sharded_config.physical_shard_shape();

    TT_FATAL(
        (global_buffer_height % shard_height == 0) and (global_buffer_width % shard_width == 0),
        "Global buffer shape must be aligned with the shard shape: requested buffer shape: ({}, {}), shard "
        "shape: ({}, {})",
        global_buffer_height,
        global_buffer_width,
        shard_height,
        shard_width);

    const auto num_shard_rows = global_buffer_height / shard_height;
    const auto num_shard_cols = global_buffer_width / shard_width;
    auto num_shards = num_shard_rows * num_shard_cols;

    // The following check needs to account for shard orientation. The scaling factor for
    // replication depends on which orientation we shard/replicate to when writing to device.
    const auto& [height_replicated, width_replicated] = sharded_config.replicated_dims();
    if (height_replicated and width_replicated) {
        // Pure replication
        num_shards *= mesh_device.num_cols() * mesh_device.num_rows();
    } else if (height_replicated or width_replicated) {
        // Replication along row or column dim.
        num_shards *=
            ((sharded_config.shard_orientation == ShardOrientation::ROW_MAJOR) * (mesh_device.num_rows()) +
             (sharded_config.shard_orientation == ShardOrientation::COL_MAJOR) * (mesh_device.num_cols()));
    }
    TT_FATAL(
        num_shards <= mesh_device.num_devices(),
        "The sharded tensor does not fit on the Mesh. Num shards in buffer {}, Num Devices {}",
        num_shards,
        mesh_device.num_devices());
}

}  // namespace

uint32_t ShardedBufferConfig::compute_datum_size_bytes() const {
    return global_size / (global_buffer_shape.height() * global_buffer_shape.width());
}

std::pair<bool, bool> ShardedBufferConfig::replicated_dims() const {
    return {shard_shape.height() == 0, shard_shape.width() == 0};
}

Shape2D ShardedBufferConfig::physical_shard_shape() const {
    const auto [shard_height, shard_width] = shard_shape;
    const auto [global_height, global_width] = global_buffer_shape;
    return Shape2D(shard_height == 0 ? global_height : shard_height, shard_width == 0 ? global_width : shard_width);
}

std::shared_ptr<MeshBuffer> MeshBuffer::create(
    const MeshBufferConfig& mesh_buffer_config,
    const DeviceLocalBufferConfig& device_local_config,
    MeshDevice* mesh_device,
    std::optional<DeviceAddr> address) {
    validate_mesh_buffer_config(mesh_buffer_config, *mesh_device);

    const DeviceAddr device_local_size = std::visit(
        ttsl::overloaded{
            [](const ReplicatedBufferConfig& c) { return c.size; },
            [](const ShardedBufferConfig& config) {
                const auto [shard_height, shard_width] = config.physical_shard_shape();
                return config.compute_datum_size_bytes() * shard_height * shard_width;
            }},
        mesh_buffer_config);

    if (mesh_device->get_view().get_devices().empty()) {
        auto mesh_buffer =
            std::shared_ptr<MeshBuffer>(new MeshBuffer(mesh_buffer_config, device_local_config, 0, 0, mesh_device));
        mesh_buffer->initialize_device_buffers();
        return mesh_buffer;
    }

    std::shared_ptr<MeshBuffer> mesh_buffer;

    // Per-core allocation path: each device allocates independently
    if (per_core_allocation::is_per_core_allocation(device_local_config.sharding_args)) {
        TT_FATAL(!address.has_value(), "Per-core allocation does not support explicit address");
        mesh_buffer = std::shared_ptr<MeshBuffer>(
            new MeshBuffer(mesh_buffer_config, device_local_config, /*address=*/0, device_local_size, mesh_device));
        // Per-core: each device allocates independently. The mesh-level lockstep allocator queries
        // device per-bank ranges at allocation time, so no explicit mirroring is needed.
        for (auto& [coord, device_buffer] : mesh_buffer->buffers_) {
            if (!mesh_device->impl().is_local(coord)) {
                continue;
            }
            auto* device = mesh_device->impl().get_device(coord);
            auto buffer = Buffer::create(
                device,
                device_local_size,
                device_local_config.page_size,
                device_local_config.buffer_type,
                device_local_config.sharding_args,
                device_local_config.bottom_up,
                device_local_config.sub_device_id);
            device_buffer = MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
        }
    } else if (!address.has_value()) {
        // In HYBRID mode, set device-level allocators on the mesh allocator so it
        // can query their per-bank ranges and avoid regions occupied on any device.
        auto* mesh_allocator = mesh_device->allocator_impl().get();
        bool is_hybrid = mesh_allocator->get_config().allocator_mode == AllocatorMode::HYBRID;
        if (is_hybrid) {
            std::vector<AllocatorImpl*> device_allocators;
            for (auto* device : mesh_device->get_view().get_devices()) {
                device_allocators.push_back(device->allocator_impl().get());
            }
            mesh_allocator->set_hybrid_device_allocators(device_allocators);
        }

        // Rely on the MeshDevice allocator to provide the address for the entire mesh buffer.
        // The address provided to the backing buffer is used as the address for the MeshBuffer object.
        std::shared_ptr<Buffer> backing_buffer = Buffer::create(
            mesh_device,
            device_local_size,
            device_local_config.page_size,
            device_local_config.buffer_type,
            device_local_config.sharding_args,
            device_local_config.bottom_up,
            device_local_config.sub_device_id);

        if (is_hybrid) {
            mesh_allocator->clear_hybrid_device_allocators();
        }

        mesh_buffer = std::shared_ptr<MeshBuffer>(new MeshBuffer(
            mesh_buffer_config, device_local_config, device_local_size, mesh_device, std::move(backing_buffer)));
        mesh_buffer->initialize_device_buffers();
    } else {
        mesh_buffer = std::shared_ptr<MeshBuffer>(
            new MeshBuffer(mesh_buffer_config, device_local_config, address.value(), device_local_size, mesh_device));
        mesh_buffer->initialize_device_buffers();
    }

    return mesh_buffer;
}

void MeshBuffer::initialize_device_buffers() {
    auto init_device_buffer_at_address = [this](const MeshCoordinate& coord) {
        std::shared_ptr<Buffer> buffer = Buffer::create(
            device()->impl().get_device(coord),
            address_,
            device_local_size_,
            device_local_config_.page_size,
            device_local_config_.buffer_type,
            device_local_config_.sharding_args,
            device_local_config_.bottom_up,
            /*sub_device_id=*/std::nullopt);  // TODO: sub_device_id is unsupported
        // For per-core allocation, propagate per-core addresses from the backing buffer.
        if (per_core_allocation::is_per_core_allocation(*buffer)) {
            TT_FATAL(
                std::holds_alternative<OwnedBufferState>(state_),
                "Per-core allocation is not supported for externally-owned MeshBuffers");
            auto& owned = std::get<OwnedBufferState>(state_);
            per_core_allocation::copy_per_core_addresses(*buffer, *owned.backing_buffer);
        }
        return buffer;
    };

    for (auto& [coord, device_buffer] : buffers_) {
        if (auto mesh_device = mesh_device_.lock(); mesh_device != nullptr) {
            if (mesh_device->impl().is_local(coord)) {
                device_buffer = MaybeRemote<std::shared_ptr<Buffer>>::local(init_device_buffer_at_address(coord));
            }
        }
    }

    // In HYBRID mode, mirror the lockstep L1 allocation into each device's lockstep allocator
    // so that per-core allocations on individual devices avoid this address range.
    // Only L1 buffers need mirroring — DRAM buffers use a separate address space.
    // Note: we check HYBRID via rtoptions rather than mesh_device->allocator_impl() because
    // allocator_impl() crashes on remote-only MeshDevices (sub_device_manager_tracker_ is null).
    if (auto mesh_device = mesh_device_.lock();
        mesh_device != nullptr && std::holds_alternative<OwnedBufferState>(state_) &&
        device_local_config_.buffer_type == BufferType::L1 &&
        MetalContext::instance(mesh_device->impl().get_context_id()).rtoptions().get_allocator_mode_hybrid()) {
        auto* backing = get_backing_buffer();
        auto alloc_size = backing->aligned_size_per_bank();
        for (const auto& [coord, device_buffer] : buffers_) {
            if (mesh_device->impl().is_local(coord)) {
                auto* device = mesh_device->impl().get_device(coord);
                device->allocator_impl()->mirror_lockstep_allocation(address_, alloc_size);
            }
        }
    }
}

bool MeshBuffer::is_allocated() const {
    if (std::holds_alternative<DeallocatedState>(state_)) {
        return false;
    }
    if (mesh_device_.lock() == nullptr) {
        return false;
    }
    return true;
}

MeshBuffer::~MeshBuffer() { deallocate(); }

MeshBuffer::MeshBuffer(MeshBuffer&& other) noexcept :
    config_(other.config_),
    device_local_config_(std::move(other.device_local_config_)),
    mesh_device_(std::move(other.mesh_device_)),
    address_(other.address_),
    device_local_size_(other.device_local_size_),
    buffers_(std::move(other.buffers_)),
    state_(std::move(other.state_)) {
    TT_ASSERT(
        other.active_event_publishers_.load(std::memory_order_acquire) == 0,
        "Cannot move a MeshBuffer with active pending-event registrations");
    // std::atomic is non-movable; transfer each slot manually.
    // Caller must guarantee no concurrent access to either object during the move.
    for (size_t i = 0; i < kMaxMeshCQs; ++i) {
        pending_event_ids_[i].value.store(
            other.pending_event_ids_[i].value.exchange(0ULL, std::memory_order_relaxed), std::memory_order_relaxed);
    }
    if (std::holds_alternative<DeallocatedState>(state_)) {
        deallocation_started_.test_and_set(std::memory_order_relaxed);
    } else {
        deallocation_started_.clear(std::memory_order_relaxed);
    }
    other.state_ = DeallocatedState{};
    other.address_ = 0;
    other.device_local_size_ = 0;
    other.deallocation_started_.test_and_set(std::memory_order_release);
}

MeshBuffer& MeshBuffer::operator=(MeshBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        TT_ASSERT(
            other.active_event_publishers_.load(std::memory_order_acquire) == 0,
            "Cannot move-assign a MeshBuffer with active pending-event registrations");
        config_ = other.config_;
        device_local_config_ = std::move(other.device_local_config_);
        mesh_device_ = std::move(other.mesh_device_);
        address_ = other.address_;
        device_local_size_ = other.device_local_size_;
        buffers_ = std::move(other.buffers_);
        state_ = std::move(other.state_);
        // std::atomic is non-movable; transfer each slot manually.
        // Caller must guarantee no concurrent access to either object during the move.
        for (size_t i = 0; i < kMaxMeshCQs; ++i) {
            pending_event_ids_[i].value.store(
                other.pending_event_ids_[i].value.exchange(0ULL, std::memory_order_relaxed), std::memory_order_relaxed);
        }
        active_event_publishers_.store(0, std::memory_order_relaxed);
        if (std::holds_alternative<DeallocatedState>(state_)) {
            deallocation_started_.test_and_set(std::memory_order_relaxed);
        } else {
            deallocation_started_.clear(std::memory_order_relaxed);
        }
        other.state_ = DeallocatedState{};
        other.address_ = 0;
        other.device_local_size_ = 0;
        other.deallocation_started_.test_and_set(std::memory_order_release);
    }
    return *this;
}

MeshBuffer::PendingEventRegistration::PendingEventRegistration(PendingEventRegistration&& other) noexcept :
    buffer_(std::exchange(other.buffer_, nullptr)) {}

MeshBuffer::PendingEventRegistration& MeshBuffer::PendingEventRegistration::operator=(
    PendingEventRegistration&& other) noexcept {
    if (this != &other) {
        release();
        buffer_ = std::exchange(other.buffer_, nullptr);
    }
    return *this;
}

MeshBuffer::PendingEventRegistration::~PendingEventRegistration() { release(); }

void MeshBuffer::PendingEventRegistration::publish(const MeshEvent& event) {
    TT_FATAL(buffer_ != nullptr, "Cannot publish an inactive MeshBuffer pending-event registration");
    buffer_->add_pending_event(event);
    release();
}

void MeshBuffer::PendingEventRegistration::release() {
    if (buffer_ != nullptr) {
        buffer_->release_pending_event_registration();
        buffer_ = nullptr;
    }
}

std::optional<MeshBuffer::PendingEventRegistration> MeshBuffer::try_acquire_pending_event_registration() const {
    if (deallocation_started_.test(std::memory_order_acquire)) {
        return std::nullopt;
    }

    active_event_publishers_.fetch_add(1, std::memory_order_acq_rel);
    if (deallocation_started_.test(std::memory_order_acquire)) {
        release_pending_event_registration();
        return std::nullopt;
    }

    PendingEventRegistration registration(this);
    return std::optional<PendingEventRegistration>(std::move(registration));
}

void MeshBuffer::release_pending_event_registration() const {
    const uint32_t previous = active_event_publishers_.fetch_sub(1, std::memory_order_release);
    TT_ASSERT(previous > 0, "MeshBuffer pending-event registration count underflow");
    if (previous == 1) {
        active_event_publishers_.notify_all();
    }
}

void MeshBuffer::wait_for_pending_event_registrations() const {
    uint32_t active = active_event_publishers_.load(std::memory_order_acquire);
    while (active != 0) {
        active_event_publishers_.wait(active, std::memory_order_acquire);
        active = active_event_publishers_.load(std::memory_order_acquire);
    }
}

void MeshBuffer::add_pending_event(const MeshEvent& event) const {
    const uint32_t cq = event.mesh_cq_id();
    TT_FATAL(cq < kMaxMeshCQs, "CQ id {} exceeds kMaxMeshCQs ({})", cq, kMaxMeshCQs);
    auto mesh_device = mesh_device_.lock();
    TT_FATAL(mesh_device != nullptr, "Cannot add a pending event after the MeshDevice has been destroyed");
    TT_FATAL(event.device() == mesh_device.get(), "MeshBuffer pending event belongs to a different MeshDevice");
    const MeshCoordinateRange full_device_range(event.device()->shape());
    TT_FATAL(
        event.device_range() == full_device_range,
        "MeshBuffer pending-event tracking requires full-mesh events. Received range {} but expected {}",
        event.device_range(),
        full_device_range);
    const uint64_t new_packed = pack_epoch_event(event.quiesce_epoch(), event.id());

    // CAS loop: only advance the stored packed value if the new one is strictly greater.
    // Within the same epoch, event IDs are monotonically increasing, so the packed
    // 64-bit value (epoch << 32 | event_id) is also monotonically increasing.
    // Across epochs, a newer epoch always dominates regardless of event_id.
    // Publishing happens before the registration's release operation. Deallocation
    // observes that release before draining these slots.
    uint64_t current = pending_event_ids_[cq].value.load(std::memory_order_relaxed);
    while (current < new_packed && !pending_event_ids_[cq].value.compare_exchange_weak(
                                       current, new_packed, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

void MeshBuffer::wait_for_pending_events() {
    auto mesh_device = mesh_device_.lock();
    if (!mesh_device) {
        return;  // MeshDevice destroyed, nothing to wait for
    }
    if (!mesh_device->is_initialized()) {
        // Device was closed (e.g. ttnn.close_device() called before Python GC runs on tensors).
        // close_impl() already flushed all outstanding work via ~FDMeshCommandQueue(), so the
        // operations behind these events have already completed.  Calling EventSynchronize() here
        // would hit Device::sysmem_manager()'s lazy-reinit path — new SystemMemoryManager starts
        // with last_completed_event=0, so the busy-spin "while (0 < event_N)" never exits.
        // Skip safely: the work is done, the device just isn't alive anymore.
        return;
    }

    // add_pending_event() accepts only full-mesh events, so reconstructing the full
    // range preserves the event synchronization contract without storing per-slot ranges.
    const MeshCoordinateRange device_range(mesh_device->shape());

    // Every publisher has released its registration before this drain starts.
    for (uint32_t cq_id = 0; cq_id < kMaxMeshCQs; ++cq_id) {
        const uint64_t packed = pending_event_ids_[cq_id].value.exchange(0ULL, std::memory_order_acquire);
        if (packed == 0) {
            continue;
        }
        const uint32_t stored_epoch = unpack_epoch(packed);
        const uint32_t event_id = unpack_event_id(packed);

        auto& mesh_cq = mesh_device->mesh_command_queue(cq_id);

        // If the parent mesh CQ has been quiesced (in_use_==false) and no new work submitted
        // on the parent mesh CQ since, then finish_and_reset_in_use() already drained all
        // outstanding work — including the work behind this event — and reset the event counters.
        // Skip safely: all work on the parent mesh CQ has already completed.
        if (!mesh_cq.in_use()) {
            continue;
        }

        // If the CQ's quiesce epoch has advanced since this event was recorded, the event
        // is from a previous quiesce cycle.  finish_and_reset_in_use() already drained all
        // work from that cycle and reset the event counters to 0.  Calling EventSynchronize
        // with the old event_id would spin forever because the new cycle's counter will
        // never reach the stale value.  Skip safely: the work is already complete.
        if (stored_epoch != mesh_cq.quiesce_epoch()) {
            continue;
        }

        EventSynchronize(MeshEvent(event_id, mesh_device.get(), cq_id, device_range, stored_epoch));
    }
}

bool MeshBuffer::has_pending_events() const {
    for (const auto& id : pending_event_ids_) {
        if (id.value.load(std::memory_order_relaxed) != 0) {
            return true;
        }
    }
    return false;
}

void MeshBuffer::deallocate() {
    if (std::holds_alternative<DeallocatedState>(state_)) {
        return;
    }

    // Close registration before waiting. A publisher either acquired before this
    // transition (and must publish/cancel before we continue) or observes the closed
    // gate and refuses to dispatch work that could reference this allocation.
    deallocation_started_.test_and_set(std::memory_order_release);
    wait_for_pending_event_registrations();

    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        // Wait for all pending operations to complete before deallocating.
        // This prevents address reuse while operations are still in-flight on other CQs.
        wait_for_pending_events();

        // Check HYBRID mode via rtoptions rather than mesh_device->allocator_impl() because:
        // 1. allocator_impl() crashes on remote-only MeshDevices (sub_device_manager_tracker_ is null).
        // 2. During teardown, device state may be partially destroyed, causing segfaults.
        if (MetalContext::instance(mesh_device->impl().get_context_id()).rtoptions().get_allocator_mode_hybrid() &&
            mesh_device->is_initialized()) {
            // Unmirror lockstep L1 allocation from each device's lockstep allocator.
            // Skip per-device unmirror if the device has been closed (default_allocator_ reset
            // by Device::close()). This can happen at process teardown when the mesh device is
            // closed before stray tensors are destroyed by the garbage collector. Mirrors the
            // device_->is_initialized() guard in Buffer::deallocate_impl().
            if (std::holds_alternative<OwnedBufferState>(state_) &&
                device_local_config_.buffer_type == BufferType::L1) {
                for (const auto& [coord, device_buffer] : buffers_) {
                    if (mesh_device->impl().is_local(coord)) {
                        auto* device = mesh_device->impl().get_device(coord);
                        if (device->is_initialized()) {
                            device->allocator_impl()->unmirror_lockstep_allocation(address_);
                        }
                    }
                }
            }

            // Per-core buffers are independently owned — drop them to trigger device-level deallocation.
            if (std::holds_alternative<ExternallyOwnedState>(state_) &&
                per_core_allocation::is_per_core_allocation(device_local_config_.sharding_args)) {
                for (auto& [coord, device_buffer] : buffers_) {
                    device_buffer = MaybeRemote<std::shared_ptr<Buffer>>::remote();
                }
            }
        }

        state_ = DeallocatedState{};
        return;
    }

    // Special handling is required if MeshDevice is already deallocated
    if (std::holds_alternative<OwnedBufferState>(state_)) {
        auto& owned_state = std::get<OwnedBufferState>(state_);
        owned_state.backing_buffer->mark_as_deallocated();
    }
    state_ = DeallocatedState{};
}

MeshDevice* MeshBuffer::device() const {
    auto device = mesh_device_.lock();
    TT_FATAL(device, "Can't get device from mesh buffer, already deallocated");
    return device.get();
}

Buffer* MeshBuffer::get_device_buffer(const MeshCoordinate& device_coord) const {
    return buffers_.at(device_coord).value().get();
}

Buffer* MeshBuffer::get_reference_buffer() const {
    for (const auto& buffer : buffers_.values()) {
        if (buffer.is_local()) {
            return buffer.value().get();
        }
    }
    TT_THROW("MeshBuffer: Tried to get reference buffer, but no local buffer found");
}

Buffer* MeshBuffer::get_backing_buffer() const {
    if (const auto* owned_state = std::get_if<OwnedBufferState>(&state_)) {
        return owned_state->backing_buffer.get();
    }
    return nullptr;
}

DeviceAddr MeshBuffer::size() const {
    return std::visit(
        ttsl::overloaded{
            [&](const ReplicatedBufferConfig& config) { return config.size; },
            [&](const ShardedBufferConfig& config) { return config.global_size; }},
        config_);
}

MeshBufferLayout MeshBuffer::global_layout() const {
    return std::holds_alternative<ReplicatedBufferConfig>(config_) ? MeshBufferLayout::REPLICATED
                                                                   : MeshBufferLayout::SHARDED;
}

const ShardedBufferConfig& MeshBuffer::global_shard_spec() const {
    TT_FATAL(
        (global_layout() == MeshBufferLayout::SHARDED),
        "Can only query the global shard spec for a sharded MeshBuffer");
    return std::get<ShardedBufferConfig>(config_);
}

uint32_t MeshBuffer::datum_size_bytes() const {
    // Limitation for now.
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query datum size for buffers sharded across the Mesh");
    return this->global_shard_spec().compute_datum_size_bytes();
}

Shape2D MeshBuffer::physical_shard_shape() const {
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query physical shard shape for buffers sharded across the Mesh");
    auto sharded_config = std::get<ShardedBufferConfig>(config_);
    return sharded_config.physical_shard_shape();
}

std::pair<bool, bool> MeshBuffer::replicated_dims() const {
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query replicated dims for buffers sharded across the Mesh");
    return this->global_shard_spec().replicated_dims();
}

AnyBuffer::AnyBuffer(std::shared_ptr<Buffer> buffer) : buffer_(buffer.get()), holder_(std::move(buffer)) {}
AnyBuffer::AnyBuffer(std::shared_ptr<MeshBuffer> buffer) :
    buffer_(buffer->get_reference_buffer()), holder_(std::move(buffer)) {}

AnyBuffer AnyBuffer::create(const tt::tt_metal::ShardedBufferConfig& config, std::optional<uint64_t> address) {
    // TODO #20966: Remove single device support and branches + dynamic_cast
    auto* mesh_device = dynamic_cast<MeshDevice*>(config.device);
    if (!mesh_device) {
        if (address.has_value()) {
            return AnyBuffer{CreateBuffer(config, *address)};
        }
        return AnyBuffer{CreateBuffer(config)};
    }
    MeshBufferConfig mesh_config = ReplicatedBufferConfig{
        .size = config.size,
    };
    DeviceLocalBufferConfig local_config{
        .page_size = config.page_size,
        .buffer_type = config.buffer_type,
        .sharding_args = BufferShardingArgs(config.shard_parameters, config.buffer_layout),
    };
    return MeshBuffer::create(mesh_config, local_config, mesh_device, address);
}

AnyBuffer AnyBuffer::create(const tt::tt_metal::InterleavedBufferConfig& config, std::optional<uint64_t> address) {
    // TODO #20966: Remove single device support and branches + dynamic_cast
    auto* mesh_device = dynamic_cast<MeshDevice*>(config.device);
    if (!mesh_device) {
        if (address.has_value()) {
            return AnyBuffer{CreateBuffer(config, *address)};
        }
        return AnyBuffer{CreateBuffer(config)};
    }
    MeshBufferConfig mesh_config = ReplicatedBufferConfig{
        .size = config.size,
    };
    DeviceLocalBufferConfig local_config{
        .page_size = config.page_size,
        .buffer_type = config.buffer_type,
    };
    return MeshBuffer::create(mesh_config, local_config, mesh_device, address);
}

Buffer* AnyBuffer::get_buffer() const { return buffer_; }

bool AnyBuffer::is_mesh_buffer() const { return get_mesh_buffer() != nullptr; }

std::shared_ptr<MeshBuffer> AnyBuffer::get_mesh_buffer() const {
    if (const auto* mesh_buffer_ptr = std::get_if<std::shared_ptr<MeshBuffer>>(&holder_)) {
        auto mesh_buffer = *mesh_buffer_ptr;
        if (mesh_buffer->is_allocated()) {
            return mesh_buffer;
        }
    }
    return nullptr;
}

}  // namespace tt::tt_metal::distributed
