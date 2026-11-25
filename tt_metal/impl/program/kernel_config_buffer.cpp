// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_config_buffer_impl.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <optional>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/memory/buffer/buffer.hpp>
#include <tt-metalium/mesh/legacy/device.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/program_preparation.hpp>
#include <tt-metalium/utils/math.hpp>
#include <tt-metalium/memory/buffer/mesh_buffer.hpp>
#include <tt-metalium/execution/mesh_command_queue.hpp>
#include <tt-metalium/mesh/mesh_device.hpp>
#include <tt-metalium/workload/mesh_workload.hpp>
#include <tt-metalium/workload/program.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/sd_mesh_command_queue.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "program_impl.hpp"
#include "slow_dispatch.hpp"

namespace tt::tt_metal::experimental {

KernelConfigBuffer::KernelConfigBuffer(std::unique_ptr<KernelConfigBufferImpl> impl) : impl_(std::move(impl)) {
    TT_FATAL(impl_ != nullptr, "A kernel config buffer needs an implementation");
}
KernelConfigBuffer::~KernelConfigBuffer() = default;
KernelConfigBuffer::KernelConfigBuffer(KernelConfigBuffer&&) noexcept = default;
KernelConfigBuffer& KernelConfigBuffer::operator=(KernelConfigBuffer&&) noexcept = default;

KernelConfigBufferImpl& KernelConfigBuffer::impl() {
    TT_FATAL(impl_ != nullptr, "Kernel config buffer was moved from");
    return *impl_;
}

const KernelConfigBufferImpl& KernelConfigBuffer::impl() const {
    TT_FATAL(impl_ != nullptr, "Kernel config buffer was moved from");
    return *impl_;
}

uint32_t KernelConfigBuffer::max_size() const { return impl().max_size(); }

CoreL1Region KernelConfigBuffer::region(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const {
    return {.address = impl().address(device_coord, core), .size = impl().size(device_coord, core)};
}

KernelConfigBufferImpl::KernelConfigBufferImpl(std::vector<Shards> shards, const distributed::MeshDevice& mesh_device) :
    shards_(std::move(shards)), mesh_device_id_(mesh_device.id()) {
    for (const distributed::MeshCoordinate& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
        if (mesh_device.impl().is_local(coord)) {
            device_ids_.emplace(coord, mesh_device.impl().get_device(coord)->id());
        }
    }
    TT_FATAL(!shards_.empty(), "A kernel config buffer needs at least one shard");
    for (size_t index = 0; index < shards_.size(); ++index) {
        const Shards& shards = shards_[index];
        const ChipId device_id = shards.buffer->device()->id();
        auto& on_device = shard_map_[device_id];
        for (const CoreCoord& core : corerange_to_cores(shards.cores)) {
            const auto [_, inserted] = on_device.emplace(core, index);
            TT_FATAL(inserted, "Kernel config buffer has two shards on device {} core {}", device_id, core);
        }
        max_size_ = std::max(max_size_, shards.size);
    }
}

const KernelConfigBufferImpl::Shards* KernelConfigBufferImpl::find(ChipId device_id, const CoreCoord& core) const {
    const auto device = shard_map_.find(device_id);
    if (device == shard_map_.end()) {
        return nullptr;
    }
    const auto shard = device->second.find(core);
    return shard != device->second.end() ? &shards_[shard->second] : nullptr;
}

const KernelConfigBufferImpl::Shards& KernelConfigBufferImpl::at(ChipId device_id, const CoreCoord& core) const {
    const Shards* shards = find(device_id, core);
    TT_FATAL(shards != nullptr, "Kernel config buffer has no shard on device {} core {}", device_id, core);
    return *shards;
}

ChipId KernelConfigBufferImpl::device_id(const distributed::MeshCoordinate& device_coord) const {
    const auto device_id = device_ids_.find(device_coord);
    TT_FATAL(
        device_id != device_ids_.end(),
        "Kernel config buffer has no shard on the device at {}, which is not on this host",
        device_coord);
    return device_id->second;
}

uint32_t KernelConfigBufferImpl::size(ChipId device_id, const CoreCoord& core) const {
    return at(device_id, core).size;
}

DeviceAddr KernelConfigBufferImpl::address(ChipId device_id, const CoreCoord& core) const {
    return per_core_allocation::get_per_core_address(*at(device_id, core).buffer, core);
}

uint32_t KernelConfigBufferImpl::size(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const {
    return size(device_id(device_coord), core);
}

DeviceAddr KernelConfigBufferImpl::address(
    const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const {
    return address(device_id(device_coord), core);
}

namespace {

// Gets the required size for the kernel config buffer on each core in a specific program on its device: the core's
// kernel binaries, aligned to the device's L1 alignment.
std::unordered_map<CoreCoord, uint32_t> get_required_buffer_sizes(IDevice& device, Program& program) {
    const uint32_t alignment = device.allocator_impl()->get_alignment(BufferType::L1);
    const uint32_t tensix_index = MetalContext::instance(program.impl().get_context_id())
                                      .hal()
                                      .get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const ProgramConfig& program_config = program.impl().get_program_config(tensix_index);
    std::unordered_map<CoreCoord, uint32_t> buffer_sizes;
    for (const auto& kernel_group : program.impl().get_kernel_groups(tensix_index)) {
        const uint32_t binaries = tt::align(kernel_text_footprint(*kernel_group, program_config), alignment);
        for (const CoreCoord& core : corerange_to_cores(kernel_group->core_ranges)) {
            buffer_sizes.emplace(core, binaries);
        }
    }
    return buffer_sizes;
}

// Allocates the shards for `program` on `device`: one per-core-allocated Buffer per shard size, each core in the one of
// its size.
std::vector<KernelConfigBufferImpl::Shards> allocate_shards(IDevice& device, Program& program, bool bottom_up) {
    std::map<uint32_t, std::vector<CoreRange>> cores_by_size;
    for (const auto& [core, buffer_size] : get_required_buffer_sizes(device, program)) {
        cores_by_size[buffer_size].emplace_back(core);
    }

    std::vector<KernelConfigBufferImpl::Shards> shards;
    std::string sizes;
    for (const auto& [buffer_size, core_list] : cores_by_size) {
        const CoreRangeSet cores = CoreRangeSet(core_list).merge_ranges();
        const uint32_t num_cores = cores.num_cores();
        // One page per core, each core allocated on its own, so the buffer takes L1 only on its cores.
        BufferShardingArgs sharding_args(
            ShardSpecBuffer(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED);
        per_core_allocation::set_per_core_allocation(sharding_args, true);
        auto buffer = BufferImpl::create(
            &device,
            static_cast<DeviceAddr>(buffer_size) * num_cores,
            buffer_size,
            BufferType::L1,
            sharding_args,
            bottom_up);
        sizes += fmt::format(" {} B x {} cores;", buffer_size, num_cores);
        shards.push_back({.cores = cores, .size = buffer_size, .buffer = std::move(buffer)});
    }
    log_debug(tt::LogMetal, "Kernel config buffer on device {}:{}", device.id(), sizes);
    return shards;
}

}  // namespace

KernelConfigBuffer CreateKernelConfigBuffer(
    distributed::MeshDevice& device, distributed::MeshWorkload& workload, bool bottom_up) {
    TT_FATAL(
        !MetalContext::instance(device.impl().get_context_id()).rtoptions().get_fast_dispatch(),
        "A kernel config buffer can only be allocated in slow dispatch mode: fast dispatch places kernel binaries in "
        "its own ring buffer");
    // Laid out as launch lays it out, so launch reuses this. Nothing to do if the caller already compiled it.
    program_preparation::compile(device, workload);

    // Each local device gets shards for the program that runs on it.
    std::vector<KernelConfigBufferImpl::Shards> shards;
    for (auto& [device_range, program] : workload.get_programs()) {
        for (const distributed::MeshCoordinate& coord : device_range) {
            if (!device.impl().is_local(coord)) {
                continue;
            }
            auto device_shards = allocate_shards(*device.impl().get_device(coord), program, bottom_up);
            std::move(device_shards.begin(), device_shards.end(), std::back_inserter(shards));
        }
    }
    TT_FATAL(!shards.empty(), "The workload has no Tensix kernels on this host's devices, so no kernel binaries");
    return KernelConfigBuffer(std::make_unique<KernelConfigBufferImpl>(std::move(shards), device));
}

void EnqueueMeshWorkload(
    distributed::MeshCommandQueue& mesh_cq,
    distributed::MeshWorkload& workload,
    const KernelConfigBuffer& buffer,
    bool blocking) {
    distributed::MeshDevice& device = *mesh_cq.device();
    TT_FATAL(
        !MetalContext::instance(device.impl().get_context_id()).rtoptions().get_fast_dispatch(),
        "A kernel config buffer can only be allocated in slow dispatch mode: fast dispatch places kernel binaries in "
        "its own ring buffer");
    TT_FATAL(
        buffer.impl().mesh_device_id() == device.id(),
        "Kernel config buffer is allocated on MeshDevice {}, but the workload is enqueued on MeshDevice {}",
        buffer.impl().mesh_device_id(),
        device.id());
    // Short-circuit for inactive MeshDevices (no-op), as distributed::EnqueueMeshWorkload does.
    if (device.get_device_ids().empty()) {
        return;
    }
    auto* sd_mesh_cq = dynamic_cast<distributed::SDMeshCommandQueue*>(&mesh_cq);
    TT_FATAL(sd_mesh_cq != nullptr, "A kernel config buffer needs a slow-dispatch mesh command queue");
    sd_mesh_cq->enqueue_mesh_workload(workload, buffer.impl(), blocking);
}

}  // namespace tt::tt_metal::experimental
