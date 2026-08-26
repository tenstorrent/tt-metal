
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
#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>
#include "device.hpp"
#include "impl/allocator/allocator.hpp"
#include "mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/debug/inspector/inspector.hpp"
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <set>

namespace per_core_allocation = tt::tt_metal::experimental::per_core_allocation;

namespace tt::tt_metal::distributed {
namespace {

// HYBRID lockstep placement on a submesh co-owned by several ranks.
//
// A lockstep address is chosen by subtracting the per-bank (per-core) reservations from the free
// list, but MeshDeviceView::get_devices() returns only local devices. On a co-owned submesh each
// co-owner therefore subtracts a different set and places one replicated buffer at a different
// address over the same physical L1, silently.
//
// The reservations are all-gathered over the co-owning ranks so every rank subtracts the same set
// and picks the same address. Co-owners also then fail together rather than one OOMing alone.

// The MPI ranks driving at least one device of `mesh_device`, sorted; empty when this rank drives
// the whole mesh.
//
// get_fabric_node_id() is a global control-plane lookup, so it answers for non-local coordinates;
// the topology mapper maps each chip to its host rank, and the control plane's global bindings map
// (mesh, host rank) to an MPI rank.
std::vector<int> compute_coowner_ranks(MeshDevice* mesh_device) {
    const auto& view = mesh_device->get_view();
    if (view.num_devices() == view.get_devices().size()) {
        return {};  // every coordinate is local: nothing is co-owned
    }

    const auto& control_plane = MetalContext::instance(mesh_device->impl().get_context_id()).get_control_plane();

    // (mesh id, host rank) -> MPI rank, inverted from the control plane's global bindings.
    std::map<std::pair<uint32_t, uint32_t>, int> rank_of_binding;
    for (const auto& [rank, binding] : control_plane.get_global_logical_bindings()) {
        rank_of_binding[{*binding.first, *binding.second}] = *rank;
    }

    std::set<int> ranks;
    std::optional<uint32_t> fabric_mesh_id;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        const auto fabric_node_id = mesh_device->get_fabric_node_id(coord);

        // Every device gathered over must belong to one fabric mesh: only devices of the same
        // mesh share an allocator address space, and a submesh straddling a boundary would pull
        // ranks from both meshes into the sub-context, where they never reach the collective
        // together.
        const uint32_t coord_mesh_id = *fabric_node_id.mesh_id;
        if (!fabric_mesh_id.has_value()) {
            fabric_mesh_id = coord_mesh_id;
        }
        TT_FATAL(
            coord_mesh_id == *fabric_mesh_id,
            "Cannot gather per-core reservations for this mesh: it spans fabric meshes {} and {} (coordinate {} is "
            "chip {} of mesh {}). Co-owner gathering assumes one fabric mesh, since only devices of the same mesh "
            "share an allocator address space.",
            *fabric_mesh_id,
            coord_mesh_id,
            coord,
            fabric_node_id.chip_id,
            coord_mesh_id);

        // Resolve through the topology mapper, the same source get_global_logical_bindings() below
        // is keyed against, so the two views cannot disagree about who owns a chip. MeshGraph
        // answers the same question from the MGD's declared host_topology instead.
        //
        // Note the chip id, not the coordinate: get_host_rank_for_chip converts to a PARENT-mesh
        // coordinate internally, while `coord` here is submesh-local. get_fabric_node_id resolves
        // that through the submesh's own handle first, which is what makes the lookup valid.
        const auto host_rank =
            control_plane.get_topology_mapper().get_host_rank_for_chip(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        TT_FATAL(
            host_rank.has_value(),
            "Cannot determine the co-owners of this mesh: chip {} of mesh {} has no host rank.",
            fabric_node_id.chip_id,
            *fabric_node_id.mesh_id);
        auto it = rank_of_binding.find({*fabric_node_id.mesh_id, **host_rank});
        TT_FATAL(
            it != rank_of_binding.end(),
            "Cannot determine the co-owners of this mesh: mesh {} host rank {} is not bound to any MPI rank.",
            *fabric_node_id.mesh_id,
            **host_rank);
        ranks.insert(it->second);
    }
    if (ranks.size() <= 1) {
        return {};
    }
    return {ranks.begin(), ranks.end()};
}

// Sub-context over `ranks`, created once and cached. create_sub_context is collective over its
// members; co-owners reach their first lockstep allocation on a submesh together, so the first
// call is made by all members at the same point.
const std::shared_ptr<multihost::DistributedContext>& coowner_context(const std::vector<int>& ranks) {
    static std::mutex cache_mutex;
    static std::map<std::vector<int>, std::shared_ptr<multihost::DistributedContext>> cache;
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(ranks);
    if (it == cache.end()) {
        auto mutable_ranks = ranks;  // create_sub_context takes a mutable span
        it = cache
                 .emplace(
                     ranks,
                     multihost::DistributedContext::get_current_world()->create_sub_context(
                         ttsl::Span<int>(mutable_ranks.data(), mutable_ranks.size())))
                 .first;
    }
    return it->second;
}

// This rank's per-bank (per-core) reservations across the devices it drives, flattened to
// [start, end, start, end, ...].
std::vector<DeviceAddr> local_per_core_ranges(
    const std::vector<AllocatorImpl*>& device_allocators, uint32_t num_banks) {
    using AllocatorID = BankManager::AllocatorDependencies::AllocatorID;
    std::vector<DeviceAddr> flat;
    for (auto* dev_alloc : device_allocators) {
        for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
            for (const auto& [start, end] : dev_alloc->get_l1_allocated_ranges(AllocatorID{bank_id + 1})) {
                flat.push_back(start);
                flat.push_back(end);
            }
        }
    }
    return flat;
}

// All-gather `local` over `ctx` and return every OTHER rank's entries as ranges.
//
// all_gather requires equal-sized contributions but rank counts differ, so gather the counts and
// pad to the maximum. Every member issues the same two collectives regardless of its own count.
std::vector<std::pair<DeviceAddr, DeviceAddr>> allgather_remote_ranges(
    const std::vector<DeviceAddr>& local, const std::vector<int>& ranks, const multihost::DistributedContext& ctx) {
    const auto world = static_cast<size_t>(*ctx.size());
    const auto my_index = static_cast<size_t>(*ctx.rank());

    uint64_t my_count = local.size();
    std::vector<uint64_t> counts(world, 0);
    ctx.all_gather(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&my_count), sizeof(my_count)),
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(counts.data()), counts.size() * sizeof(uint64_t)));

    const uint64_t max_count = *std::max_element(counts.begin(), counts.end());
    if (max_count == 0) {
        return {};
    }

    std::vector<DeviceAddr> padded(max_count, 0);
    std::copy(local.begin(), local.end(), padded.begin());
    std::vector<DeviceAddr> gathered(world * max_count, 0);
    ctx.all_gather(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(padded.data()), padded.size() * sizeof(DeviceAddr)),
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(gathered.data()), gathered.size() * sizeof(DeviceAddr)));

    std::vector<std::pair<DeviceAddr, DeviceAddr>> remote;
    for (size_t r = 0; r < world; r++) {
        if (r == my_index) {
            continue;  // the caller already has its own, gathered from live allocators
        }
        for (uint64_t i = 0; i + 1 < counts[r]; i += 2) {
            const DeviceAddr start = gathered[r * max_count + i];
            const DeviceAddr end = gathered[r * max_count + i + 1];
            if (end > start) {
                remote.emplace_back(start, end);
            }
        }
    }
    log_debug(
        tt::LogMetal,
        "[hybrid-coowner] rank {} contributed {} range(s), received {} from the other {} co-owner(s) of this mesh",
        *multihost::DistributedContext::get_current_world()->rank(),
        local.size() / 2,
        remote.size(),
        ranks.size() - 1);
    return remote;
}

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
        Inspector::mesh_buffer_allocated(mesh_buffer.get());
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
            device_allocators.reserve(mesh_device->get_view().num_devices());
            for (auto* device : mesh_device->get_view().get_devices()) {
                device_allocators.push_back(device->allocator_impl().get());
            }
            mesh_allocator->set_hybrid_device_allocators(device_allocators);

            // The loop above sees only local devices, so trade per-bank reservations with the
            // co-owners and let each subtract the same occupied set. No collective on a mesh this
            // rank drives alone. Done here rather than in AllocatorImpl because allocate_buffer()
            // holds the allocator mutex, under which a collective must not run.
            if (device_local_config.buffer_type == BufferType::L1) {
                const auto coowners = compute_coowner_ranks(mesh_device);
                if (!coowners.empty()) {
                    const auto& ctx = coowner_context(coowners);
                    const uint32_t num_banks = mesh_allocator->get_num_banks(BufferType::L1);
                    mesh_allocator->set_hybrid_remote_occupied_ranges(
                        allgather_remote_ranges(local_per_core_ranges(device_allocators, num_banks), coowners, *ctx));
                }
            }
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
            mesh_allocator->clear_hybrid_remote_occupied_ranges();
        }

        mesh_buffer = std::shared_ptr<MeshBuffer>(new MeshBuffer(
            mesh_buffer_config, device_local_config, device_local_size, mesh_device, std::move(backing_buffer)));
        mesh_buffer->initialize_device_buffers();
    } else {
        mesh_buffer = std::shared_ptr<MeshBuffer>(
            new MeshBuffer(mesh_buffer_config, device_local_config, address.value(), device_local_size, mesh_device));
        mesh_buffer->initialize_device_buffers();
    }

    Inspector::mesh_buffer_allocated(mesh_buffer.get());
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
    config_((Inspector::mesh_buffer_deallocated(&other), other.config_)),
    device_local_config_(std::move(other.device_local_config_)),
    mesh_device_(std::move(other.mesh_device_)),
    address_(other.address_),
    device_local_size_(other.device_local_size_),
    buffers_(std::move(other.buffers_)),
    state_(std::move(other.state_)) {
    other.state_ = DeallocatedState{};
    other.address_ = 0;
    other.device_local_size_ = 0;
    Inspector::mesh_buffer_allocated(this);
}

MeshBuffer& MeshBuffer::operator=(MeshBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        Inspector::mesh_buffer_deallocated(&other);
        config_ = other.config_;
        device_local_config_ = std::move(other.device_local_config_);
        mesh_device_ = std::move(other.mesh_device_);
        address_ = other.address_;
        device_local_size_ = other.device_local_size_;
        buffers_ = std::move(other.buffers_);
        state_ = std::move(other.state_);

        other.state_ = DeallocatedState{};
        other.address_ = 0;
        other.device_local_size_ = 0;
        Inspector::mesh_buffer_allocated(this);
    }
    return *this;
}

void MeshBuffer::deallocate() {
    // Guard against double reporting to Inspector if deallocate() was called explicitly and then again in the
    // destructor.
    if (!std::holds_alternative<DeallocatedState>(state_)) {
        Inspector::mesh_buffer_deallocated(this);
    }

    auto mesh_device = mesh_device_.lock();
    if (mesh_device) {
        // Check HYBRID mode via rtoptions rather than mesh_device->allocator_impl() because:
        // 1. allocator_impl() crashes on remote-only MeshDevices (sub_device_manager_tracker_ is null).
        // 2. During teardown, device state may be partially destroyed, causing segfaults.
        if (MetalContext::instance(mesh_device->impl().get_context_id()).rtoptions().get_allocator_mode_hybrid()) {
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
