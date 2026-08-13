
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include "impl/buffers/buffer_impl.hpp"
#include <mesh_buffer.hpp>
#include <mesh_coord.hpp>
#include <tt_stl/overloaded.hpp>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>
#include "device.hpp"
#include "impl/allocator/allocator.hpp"
#include "mesh_device_impl.hpp"
#include "mesh_buffer_impl.hpp"
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
            std::make_shared<MeshBuffer>(MeshBufferImpl(mesh_buffer_config, device_local_config, 0, 0, mesh_device));
        mesh_buffer->impl().initialize_device_buffers(*mesh_buffer);
        return mesh_buffer;
    }

    std::shared_ptr<MeshBuffer> mesh_buffer;

    // Per-core allocation path: each device allocates independently
    if (per_core_allocation::is_per_core_allocation(device_local_config.sharding_args)) {
        TT_FATAL(!address.has_value(), "Per-core allocation does not support explicit address");
        mesh_buffer = std::make_shared<MeshBuffer>(
            MeshBufferImpl(mesh_buffer_config, device_local_config, /*address=*/0, device_local_size, mesh_device));
        // Per-core: each device allocates independently. The mesh-level lockstep allocator queries
        // device per-bank ranges at allocation time, so no explicit mirroring is needed.
        for (auto& [coord, device_buffer] : mesh_buffer->impl().buffers_) {
            if (!mesh_device->impl().is_local(coord)) {
                continue;
            }
            auto* device = mesh_device->impl().get_device(coord);
            auto buffer = BufferImpl::create(
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
        auto* mesh_allocator = mesh_device->allocator_impl().get();
        bool is_hybrid = mesh_allocator->get_config().allocator_mode == AllocatorMode::HYBRID;
        if (is_hybrid) {
            std::vector<AllocatorImpl*> device_allocators;
            device_allocators.reserve(mesh_device->get_view().num_devices());
            for (auto* device : mesh_device->get_view().get_devices()) {
                device_allocators.push_back(device->allocator_impl().get());
            }
            mesh_allocator->set_hybrid_device_allocators(device_allocators);
        }

        std::shared_ptr<Buffer> backing_buffer = BufferImpl::create(
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

        mesh_buffer = std::make_shared<MeshBuffer>(MeshBufferImpl(
            mesh_buffer_config, device_local_config, device_local_size, mesh_device, std::move(backing_buffer)));
        mesh_buffer->impl().initialize_device_buffers(*mesh_buffer);
    } else {
        mesh_buffer = std::make_shared<MeshBuffer>(
            MeshBufferImpl(mesh_buffer_config, device_local_config, address.value(), device_local_size, mesh_device));
        mesh_buffer->impl().initialize_device_buffers(*mesh_buffer);
    }

    return mesh_buffer;
}

MeshBuffer::MeshBuffer(MeshBufferImpl impl) : impl_(std::make_unique<MeshBufferImpl>(std::move(impl))) {}

MeshBufferImpl& MeshBuffer::impl() {
    TT_FATAL(impl_ != nullptr, "MeshBuffer is in a moved-from state.");
    return *impl_;
}

const MeshBufferImpl& MeshBuffer::impl() const {
    TT_FATAL(impl_ != nullptr, "MeshBuffer is in a moved-from state.");
    return *impl_;
}

void MeshBufferImpl::initialize_device_buffers(MeshBuffer& self) {
    auto init_device_buffer_at_address = [&](const MeshCoordinate& coord) {
        std::shared_ptr<Buffer> buffer = BufferImpl::create(
            self.device()->impl().get_device(coord),
            address_,
            device_local_size_,
            device_local_config_.page_size,
            device_local_config_.buffer_type,
            device_local_config_.sharding_args,
            device_local_config_.bottom_up,
            /*sub_device_id=*/std::nullopt);  // TODO: sub_device_id is unsupported
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

    if (auto mesh_device = mesh_device_.lock();
        mesh_device != nullptr && std::holds_alternative<OwnedBufferState>(state_) &&
        device_local_config_.buffer_type == BufferType::L1 &&
        MetalContext::instance(mesh_device->impl().get_context_id()).rtoptions().get_allocator_mode_hybrid()) {
        auto* backing = self.get_backing_buffer();
        auto alloc_size = backing->aligned_size_per_bank();
        for (const auto& [coord, device_buffer] : buffers_) {
            if (mesh_device->impl().is_local(coord)) {
                auto* device = mesh_device->impl().get_device(coord);
                device->allocator_impl()->mirror_lockstep_allocation(address_, alloc_size);
            }
        }
    }
}

bool MeshBufferImpl::is_allocated() const {
    if (std::holds_alternative<DeallocatedState>(state_)) {
        return false;
    }
    if (mesh_device_.lock() == nullptr) {
        return false;
    }
    return true;
}

MeshBuffer::~MeshBuffer() {
    if (impl_) {
        impl_->deallocate();
    }
}

MeshBuffer::MeshBuffer(MeshBuffer&& other) noexcept = default;

MeshBuffer& MeshBuffer::operator=(MeshBuffer&& other) noexcept {
    if (this != &other) {
        if (impl_) {
            impl_->deallocate();
        }
        impl_ = std::move(other.impl_);
    }
    return *this;
}

void MeshBufferImpl::deallocate() {
    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        if (MetalContext::instance(mesh_device->impl().get_context_id()).rtoptions().get_allocator_mode_hybrid()) {
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

    if (std::holds_alternative<OwnedBufferState>(state_)) {
        auto& owned_state = std::get<OwnedBufferState>(state_);
        owned_state.backing_buffer->impl().mark_as_deallocated();
    }
    state_ = DeallocatedState{};
}

MeshDevice* MeshBuffer::device() const {
    auto device = impl_->mesh_device_.lock();
    TT_FATAL(device, "Can't get device from mesh buffer, already deallocated");
    return device.get();
}

DeviceAddr MeshBuffer::device_local_size() const { return impl_->device_local_size_; }
DeviceAddr MeshBuffer::address() const { return impl_->address_; }
const MeshBufferConfig& MeshBuffer::global_config() const { return impl_->config_; }
const DeviceLocalBufferConfig& MeshBuffer::device_local_config() const { return impl_->device_local_config_; }
uint32_t MeshBuffer::page_size() const { return impl_->device_local_config_.page_size; }
uint32_t MeshBuffer::num_pages() const { return page_size() == 0 ? 0 : device_local_size() / page_size(); }
DeviceAddr MeshBuffer::size() const { return impl_->size(); }
const ShardedBufferConfig& MeshBuffer::global_shard_spec() const { return impl_->global_shard_spec(); }

Buffer* MeshBuffer::get_device_buffer(const MeshCoordinate& device_coord) const {
    return impl_->buffers_.at(device_coord).value().get();
}

Buffer* MeshBuffer::get_reference_buffer() const {
    for (const auto& buffer : impl_->buffers_.values()) {
        if (buffer.is_local()) {
            return buffer.value().get();
        }
    }
    TT_THROW("MeshBuffer: Tried to get reference buffer, but no local buffer found");
}

Buffer* MeshBuffer::get_backing_buffer() const {
    if (const auto* owned_state = std::get_if<MeshBufferImpl::OwnedBufferState>(&impl_->state_)) {
        return owned_state->backing_buffer.get();
    }
    return nullptr;
}

DeviceAddr MeshBufferImpl::size() const {
    return std::visit(
        ttsl::overloaded{
            [&](const ReplicatedBufferConfig& config) { return config.size; },
            [&](const ShardedBufferConfig& config) { return config.global_size; }},
        config_);
}

MeshBufferLayout MeshBufferImpl::global_layout() const {
    return std::holds_alternative<ReplicatedBufferConfig>(config_) ? MeshBufferLayout::REPLICATED
                                                                   : MeshBufferLayout::SHARDED;
}

MeshBufferLayout MeshBuffer::global_layout() const { return impl_->global_layout(); }

const ShardedBufferConfig& MeshBufferImpl::global_shard_spec() const {
    TT_FATAL(
        (global_layout() == MeshBufferLayout::SHARDED),
        "Can only query the global shard spec for a sharded MeshBuffer");
    return std::get<ShardedBufferConfig>(config_);
}

uint32_t MeshBufferImpl::datum_size_bytes() const {
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query datum size for buffers sharded across the Mesh");
    return this->global_shard_spec().compute_datum_size_bytes();
}

Shape2D MeshBufferImpl::physical_shard_shape() const {
    TT_FATAL(
        this->global_layout() == MeshBufferLayout::SHARDED,
        "Can only query physical shard shape for buffers sharded across the Mesh");
    auto sharded_config = std::get<ShardedBufferConfig>(config_);
    return sharded_config.physical_shard_shape();
}

std::pair<bool, bool> MeshBufferImpl::replicated_dims() const {
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
        if (mesh_buffer->impl().is_allocated()) {
            return mesh_buffer;
        }
    }
    return nullptr;
}

}  // namespace tt::tt_metal::distributed
