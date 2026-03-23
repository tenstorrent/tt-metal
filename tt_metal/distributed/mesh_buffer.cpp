
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <mesh_buffer.hpp>
#include <mesh_coord.hpp>
#include <tt_stl/overloaded.hpp>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/allocator_mode.hpp>
#include "device.hpp"
#include "impl/allocator/allocator.hpp"
#include "mesh_device_impl.hpp"

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

    // Per-core allocation path: each device allocates independently
    if (device_local_config.sharding_args.per_core_allocation()) {
        TT_FATAL(!address.has_value(), "Per-core allocation does not support explicit address");
        mesh_buffer = std::shared_ptr<MeshBuffer>(
            new MeshBuffer(mesh_buffer_config, device_local_config, /*address=*/0, device_local_size, mesh_device));
        mesh_buffer->state_ = MeshBuffer::PerCoreOwnedState{};
        mesh_buffer->initialize_device_buffers_per_core();
    } else if (!address.has_value()) {
        // In HYBRID mode, gather device-level allocators so the mesh lockstep allocator
        // can query their per-bank ranges and avoid regions occupied on any device.
        std::vector<AllocatorImpl*> device_allocators;
        if (mesh_device->allocator_impl()->get_config().allocator_mode == AllocatorMode::HYBRID) {
            for (auto* device : mesh_device->get_view().get_devices()) {
                device_allocators.push_back(device->allocator_impl().get());
            }
        }

        // Rely on the MeshDevice allocator to provide the address for the entire mesh buffer.
        std::shared_ptr<Buffer> backing_buffer = Buffer::create(
            mesh_device,
            device_local_size,
            device_local_config.page_size,
            device_local_config.buffer_type,
            device_local_config.sharding_args,
            device_local_config.bottom_up,
            device_local_config.sub_device_id,
            device_allocators);

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

std::shared_ptr<MeshBuffer> MeshBuffer::create_on_single_device(
    const MeshBufferConfig& mesh_buffer_config,
    const DeviceLocalBufferConfig& device_local_config,
    MeshDevice* mesh_device,
    const MeshCoordinate& coord) {
    const DeviceAddr device_local_size = std::visit(
        tt::stl::overloaded{
            [](const ReplicatedBufferConfig& c) { return c.size; },
            [](const ShardedBufferConfig& config) {
                const auto [shard_height, shard_width] = config.physical_shard_shape();
                return config.compute_datum_size_bytes() * shard_height * shard_width;
            }},
        mesh_buffer_config);

    // Create a non-owning MeshBuffer — each device buffer will own its own allocation.
    auto mesh_buffer = std::shared_ptr<MeshBuffer>(
        new MeshBuffer(mesh_buffer_config, device_local_config, /*address=*/0, device_local_size, mesh_device));

    if (device_local_config.sharding_args.per_core_allocation()) {
        mesh_buffer->state_ = MeshBuffer::PerCoreOwnedState{};
    }

    // Only allocate on the target device.
    TT_FATAL(mesh_device->impl().is_local(coord), "Target device coordinate must be local");
    auto* device = mesh_device->impl().get_device(coord);
    auto buffer = Buffer::create(
        device,
        device_local_size,
        device_local_config.page_size,
        device_local_config.buffer_type,
        device_local_config.sharding_args,
        device_local_config.bottom_up,
        device_local_config.sub_device_id);

    mesh_buffer->buffers_.at(coord) = MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
    return mesh_buffer;
}

void MeshBuffer::initialize_device_buffers() {
    auto mesh_device = mesh_device_.lock();
    TT_FATAL(mesh_device, "MeshDevice expired");

    for (auto& [coord, device_buffer] : buffers_) {
        if (mesh_device->impl().is_local(coord)) {
            auto buffer = Buffer::create(
                mesh_device->impl().get_device(coord),
                address_,
                device_local_size_,
                device_local_config_.page_size,
                device_local_config_.buffer_type,
                device_local_config_.sharding_args,
                device_local_config_.bottom_up,
                /*sub_device_id=*/std::nullopt);  // TODO: sub_device_id is unsupported
            device_buffer = MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
        }
    }

    // In HYBRID mode, mirror the lockstep allocation into each device's lockstep allocator
    // so that per-core allocations on individual devices avoid this address range.
    if (std::holds_alternative<OwnedBufferState>(state_)) {
        auto* mesh_allocator = mesh_device->allocator_impl().get();
        if (mesh_allocator->get_config().allocator_mode == AllocatorMode::HYBRID) {
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
}

void MeshBuffer::initialize_device_buffers_per_core() {
    auto mesh_device = mesh_device_.lock();
    TT_FATAL(mesh_device, "MeshDevice expired");

    for (auto& [coord, device_buffer] : buffers_) {
        if (!mesh_device->impl().is_local(coord)) {
            continue;
        }

        // Create an OWNING buffer on this device — each device's own allocator handles per-core allocation.
        // The mesh-level lockstep allocator queries device per-bank ranges at allocation time,
        // so no explicit mirroring is needed here.
        auto* device = mesh_device->impl().get_device(coord);
        auto buffer = Buffer::create(
            device,
            device_local_size_,
            device_local_config_.page_size,
            device_local_config_.buffer_type,
            device_local_config_.sharding_args,
            device_local_config_.bottom_up,
            device_local_config_.sub_device_id);

        device_buffer = MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
    }
}

DeviceAddr MeshBuffer::per_core_address(const MeshCoordinate& device_coord, const CoreCoord& core) const {
    auto* buffer = get_device_buffer(device_coord);
    TT_FATAL(buffer->per_core_allocation(), "Buffer does not use per-core allocation");
    return buffer->per_core_address(core);
}

bool MeshBuffer::is_per_core_allocation() const { return std::holds_alternative<PerCoreOwnedState>(state_); }

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

void MeshBuffer::deallocate() {
    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        // Unmirror cross-allocator state before deallocation
        auto* mesh_allocator = mesh_device->allocator_impl().get();
        bool is_hybrid = mesh_allocator->get_config().allocator_mode == AllocatorMode::HYBRID;

        if (std::holds_alternative<OwnedBufferState>(state_) && is_hybrid) {
            // Unmirror lockstep allocation from each device's lockstep allocator
            for (const auto& [coord, device_buffer] : buffers_) {
                if (mesh_device->impl().is_local(coord)) {
                    auto* device = mesh_device->impl().get_device(coord);
                    device->allocator_impl()->unmirror_lockstep_allocation(address_);
                }
            }
        }

        if (std::holds_alternative<PerCoreOwnedState>(state_)) {
            // Per-core buffers are independently owned — drop them to trigger device-level deallocation.
            for (auto& [coord, device_buffer] : buffers_) {
                device_buffer = MaybeRemote<std::shared_ptr<Buffer>>::remote();
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

DeviceAddr MeshBuffer::address() const {
    TT_FATAL(
        !std::holds_alternative<PerCoreOwnedState>(state_),
        "MeshBuffer::address() is not valid for per-core allocated buffers, use per_core_address() instead");
    return address_;
}

DeviceAddr MeshBuffer::per_core_address(const CoreCoord& core) const {
    auto* buffer = get_reference_buffer();
    TT_FATAL(buffer->per_core_allocation(), "Buffer does not use per-core allocation");
    return buffer->per_core_address(core);
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
