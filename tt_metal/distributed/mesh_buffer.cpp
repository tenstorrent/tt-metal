
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <mesh_buffer.hpp>
#include <mesh_coord.hpp>
#include <tt_stl/overloaded.hpp>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include "device.hpp"
#include "mesh_buffer_impl.hpp"
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

// ---------------------------------------------------------------------------
// MeshBuffer::Impl method definitions
// ---------------------------------------------------------------------------

void MeshBuffer::Impl::initialize_device_buffers() {
    for (auto& [coord, device_buffer] : buffers_) {
        auto mesh_device = mesh_device_.lock();
        if (mesh_device == nullptr) {
            continue;
        }
        if (!mesh_device->impl().is_local(coord)) {
            continue;
        }
        std::shared_ptr<Buffer> buffer = Buffer::create(
            mesh_device->impl().get_device(coord),
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
        device_buffer = MaybeRemote<std::shared_ptr<Buffer>>::local(std::move(buffer));
    }
}

void MeshBuffer::Impl::deallocate() {
    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
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

// ---------------------------------------------------------------------------
// MeshBuffer
// ---------------------------------------------------------------------------

MeshBuffer::MeshBuffer(std::unique_ptr<Impl> impl) : pimpl_(std::move(impl)) {}

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
        auto impl = std::make_unique<Impl>(
            mesh_buffer_config, device_local_config, DeviceAddr{0}, DeviceAddr{0}, mesh_device, /*empty_tag=*/true);
        impl->initialize_device_buffers();
        return std::shared_ptr<MeshBuffer>(new MeshBuffer(std::move(impl)));
    }

    std::unique_ptr<Impl> impl;
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

        impl = std::make_unique<Impl>(
            mesh_buffer_config, device_local_config, device_local_size, mesh_device, std::move(backing_buffer));
    } else {
        impl = std::make_unique<Impl>(
            mesh_buffer_config, device_local_config, address.value(), device_local_size, mesh_device);
    }

    impl->initialize_device_buffers();
    return std::shared_ptr<MeshBuffer>(new MeshBuffer(std::move(impl)));
}

MeshBuffer::~MeshBuffer() {
    if (pimpl_) {
        pimpl_->deallocate();
    }
}

MeshBuffer::MeshBuffer(MeshBuffer&&) noexcept = default;

MeshBuffer& MeshBuffer::operator=(MeshBuffer&& other) noexcept {
    if (this != &other) {
        if (pimpl_) {
            pimpl_->deallocate();
        }
        pimpl_ = std::move(other.pimpl_);
    }
    return *this;
}

bool MeshBuffer::is_allocated() const {
    if (!pimpl_) {
        return false;
    }
    if (std::holds_alternative<MeshBuffer::Impl::DeallocatedState>(pimpl_->state_)) {
        return false;
    }
    if (pimpl_->mesh_device_.lock() == nullptr) {
        return false;
    }
    return true;
}

void MeshBuffer::deallocate() { pimpl_->deallocate(); }

MeshDevice* MeshBuffer::device() const {
    auto device = pimpl_->mesh_device_.lock();
    TT_FATAL(device, "Can't get device from mesh buffer, already deallocated");
    return device.get();
}

DeviceAddr MeshBuffer::size() const {
    return std::visit(
        tt::stl::overloaded{
            [&](const ReplicatedBufferConfig& config) { return config.size; },
            [&](const ShardedBufferConfig& config) { return config.global_size; }},
        pimpl_->config_);
}

DeviceAddr MeshBuffer::device_local_size() const { return pimpl_->device_local_size_; }

DeviceAddr MeshBuffer::address() const { return pimpl_->address_; }

MeshBufferLayout MeshBuffer::global_layout() const {
    return std::holds_alternative<ReplicatedBufferConfig>(pimpl_->config_) ? MeshBufferLayout::REPLICATED
                                                                           : MeshBufferLayout::SHARDED;
}

const MeshBufferConfig& MeshBuffer::global_config() const { return pimpl_->config_; }

const ShardedBufferConfig& MeshBuffer::global_shard_spec() const {
    TT_FATAL(
        (global_layout() == MeshBufferLayout::SHARDED),
        "Can only query the global shard spec for a sharded MeshBuffer");
    return std::get<ShardedBufferConfig>(pimpl_->config_);
}

const DeviceLocalBufferConfig& MeshBuffer::device_local_config() const { return pimpl_->device_local_config_; }

Buffer* MeshBuffer::get_device_buffer(const MeshCoordinate& device_coord) const {
    return pimpl_->buffers_.at(device_coord).value().get();
}

Buffer* MeshBuffer::get_reference_buffer() const {
    for (const auto& buffer : pimpl_->buffers_.values()) {
        if (buffer.is_local()) {
            return buffer.value().get();
        }
    }
    TT_THROW("MeshBuffer: Tried to get reference buffer, but no local buffer found");
}

Buffer* MeshBuffer::get_backing_buffer() const {
    if (const auto* owned_state = std::get_if<Impl::OwnedBufferState>(&pimpl_->state_)) {
        return owned_state->backing_buffer.get();
    }
    return nullptr;
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
    auto sharded_config = std::get<ShardedBufferConfig>(pimpl_->config_);
    return sharded_config.physical_shard_shape();
}

std::pair<bool, bool> MeshBuffer::replicated_dims() const {
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query replicated dims for buffers sharded across the Mesh");
    return this->global_shard_spec().replicated_dims();
}

uint32_t MeshBuffer::page_size() const { return pimpl_->device_local_config_.page_size; }

uint32_t MeshBuffer::num_pages() const { return pimpl_->device_local_size_ / pimpl_->device_local_config_.page_size; }

// ---------------------------------------------------------------------------
// AnyBuffer
// ---------------------------------------------------------------------------

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
