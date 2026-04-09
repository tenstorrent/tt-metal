
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
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include "device.hpp"
#include "mesh_device_impl.hpp"

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
        tt::stl::overloaded{
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
    if (!address.has_value()) {
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

        mesh_buffer = std::shared_ptr<MeshBuffer>(new MeshBuffer(
            mesh_buffer_config, device_local_config, device_local_size, mesh_device, std::move(backing_buffer)));
    } else {
        mesh_buffer = std::shared_ptr<MeshBuffer>(
            new MeshBuffer(mesh_buffer_config, device_local_config, address.value(), device_local_size, mesh_device));
    }

    mesh_buffer->initialize_device_buffers();

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
    // std::atomic is non-movable; transfer each slot manually.
    // Caller must guarantee no concurrent access to either object during the move.
    for (size_t i = 0; i < kMaxMeshCQs; ++i) {
        pending_event_ids_[i].store(
            other.pending_event_ids_[i].exchange(0, std::memory_order_relaxed),
            std::memory_order_relaxed);
    }
    // The moved-to object is freshly constructed — deallocation is not in progress.
    deallocation_in_progress_.store(false, std::memory_order_relaxed);
    other.state_ = DeallocatedState{};
    other.address_ = 0;
    other.device_local_size_ = 0;
}

MeshBuffer& MeshBuffer::operator=(MeshBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
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
            pending_event_ids_[i].store(
                other.pending_event_ids_[i].exchange(0, std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        // After move-assign, deallocation is not in progress on this object.
        deallocation_in_progress_.store(false, std::memory_order_relaxed);
        other.state_ = DeallocatedState{};
        other.address_ = 0;
        other.device_local_size_ = 0;
    }
    return *this;
}

void MeshBuffer::add_pending_event(const MeshEvent& event) {
    const uint32_t cq = event.mesh_cq_id();
    TT_FATAL(cq < kMaxMeshCQs, "CQ id {} exceeds kMaxMeshCQs ({})", cq, kMaxMeshCQs);
    const uint32_t new_id = event.id();

    // CAS loop: only advance the stored ID if new_id is strictly greater.
    // memory_order_seq_cst on success participates in the total order with the
    // seq_cst drain in wait_for_pending_events() and the seq_cst store of
    // deallocation_in_progress_ in deallocate(), closing the add/drain race window.
    // See proof below.
    uint32_t current = pending_event_ids_[cq].load(std::memory_order_relaxed);
    while (current < new_id &&
           !pending_event_ids_[cq].compare_exchange_weak(
               current, new_id, std::memory_order_seq_cst, std::memory_order_relaxed)) {
    }

    // Deallocation race guard.
    //
    // Without this check, the following window exists:
    //   Thread A: enqueue_record_event_to_host() → [api_mutex released] → add_pending_event(E)
    //   Thread B: deallocate() → sets deallocation_in_progress_ → drain slots (gets 0) → frees buffer
    //   Thread A: stores E → returns → device still executing → buffer address reused → corruption
    //
    // Closed by seq_cst total order:
    //   Thread B program order: deallocation_in_progress_.store(true, seq_cst) → exchange(0, seq_cst)
    //   Thread A program order: CAS(seq_cst) → load(deallocation_in_progress_, seq_cst)
    //
    //   For the bad case (exchange sees 0 = CAS not yet done):
    //     In seq_cst total order: dealloc_store → exchange → [CAS] → load
    //     Since dealloc_store precedes CAS precedes load in total order, the seq_cst load MUST
    //     observe true → Thread A self-synchronizes before returning. ✓
    //
    //   For the good case (exchange sees new_id = CAS already done):
    //     Thread B waits. Thread A's load may also see true (harmless double-wait). ✓
    if (deallocation_in_progress_.load(std::memory_order_seq_cst)) {
        EventSynchronize(event);
    }
}

void MeshBuffer::wait_for_pending_events() {
    auto mesh_device = mesh_device_.lock();
    if (!mesh_device) {
        return;  // MeshDevice destroyed, nothing to wait for
    }

    // For the device_operation.hpp dispatch path, enqueue_record_event_to_host() is called
    // without an explicit range, so it always targets the full mesh.
    const MeshCoordinateRange device_range(mesh_device->shape());

    // Drain each slot with seq_cst to participate in the total order with the
    // seq_cst CAS in add_pending_event and the seq_cst store of deallocation_in_progress_
    // in deallocate(). See proof in add_pending_event.
    for (uint32_t cq_id = 0; cq_id < kMaxMeshCQs; ++cq_id) {
        const uint32_t event_id =
            pending_event_ids_[cq_id].exchange(0, std::memory_order_seq_cst);
        if (event_id == 0) {
            continue;
        }
        EventSynchronize(MeshEvent(event_id, mesh_device.get(), cq_id, device_range));
    }
}

bool MeshBuffer::has_pending_events() const {
    for (const auto& id : pending_event_ids_) {
        if (id.load(std::memory_order_relaxed) != 0) {
            return true;
        }
    }
    return false;
}

void MeshBuffer::deallocate() {
    if (std::holds_alternative<DeallocatedState>(state_)) {
        return;
    }

    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        // Signal that deallocation is in progress BEFORE draining pending_event_ids_.
        // This participates in the seq_cst total order with the CAS in add_pending_event:
        // any add_pending_event that runs after this store will observe true and
        // self-synchronize, closing the window between drain and a late add_pending_event call.
        deallocation_in_progress_.store(true, std::memory_order_seq_cst);

        // Wait for all pending operations to complete before deallocating.
        // This prevents address reuse while operations are still in-flight on other CQs.
        wait_for_pending_events();

        // Now safe to deallocate the backing buffer
        if (std::holds_alternative<OwnedBufferState>(state_)) {
            auto& owned_state = std::get<OwnedBufferState>(state_);
            // Release the backing buffer which will return the address to the allocator
            owned_state.backing_buffer.reset();
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
        tt::stl::overloaded{
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
