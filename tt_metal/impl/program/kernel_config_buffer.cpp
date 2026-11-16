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
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/program_preparation.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/sd_mesh_command_queue.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/buffer_impl.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "program_impl.hpp"
#include "slow_dispatch.hpp"

namespace tt::tt_metal::experimental {

namespace {

uint32_t tensix_index(const Program& program) {
    return MetalContext::instance(program.impl().get_context_id())
        .hal()
        .get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

void check_no_tensix_semaphores(Program& program) {
    for (const Semaphore& semaphore : program.impl().semaphores()) {
        TT_FATAL(
            semaphore.core_type() != CoreType::WORKER,
            "A program run from a kernel config buffer cannot have Tensix program semaphores (semaphore {} on {}): "
            "each core's block, and so each semaphore, sits at a different address, but a kernel signals a peer's "
            "semaphore at its own address. Use a global semaphore instead.",
            semaphore.id(),
            semaphore.core_range_set());
    }
}

void check_slow_dispatch(distributed::MeshDevice& device) {
    TT_FATAL(
        !MetalContext::instance(device.impl().get_context_id()).rtoptions().get_fast_dispatch(),
        "A kernel config buffer needs slow dispatch: fast dispatch places kernel-config blocks in its own ring "
        "buffer");
}

}  // namespace

KernelConfigBuffer::KernelConfigBuffer(std::shared_ptr<KernelConfigBufferImpl> impl) : impl_(std::move(impl)) {}

const CoreRangeSet& KernelConfigBuffer::cores() const { return impl_->cores(); }

uint32_t KernelConfigBuffer::size() const { return impl_->size(); }

uint32_t KernelConfigBuffer::size(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const {
    return impl_->size(device_coord, core);
}

DeviceAddr KernelConfigBuffer::address(const distributed::MeshCoordinate& device_coord, const CoreCoord& core) const {
    return impl_->address(device_coord, core);
}

void KernelConfigBuffer::deallocate() { impl_->deallocate(); }

KernelConfigBufferImpl::KernelConfigBufferImpl(
    std::vector<Shards> shards, bool lockstep_address, const distributed::MeshDevice* mesh_device) :
    shards_(std::move(shards)), mesh_device_(mesh_device), lockstep_address_(lockstep_address) {
    TT_FATAL(!shards_.empty(), "A kernel config buffer needs at least one shard");
    for (size_t index = 0; index < shards_.size(); ++index) {
        const Shards& shards = shards_[index];
        const ChipId device_id = shards.buffer->device()->id();
        auto& on_device = shards_on_device_[device_id];
        for (const CoreCoord& core : corerange_to_cores(shards.cores)) {
            const auto [_, inserted] = on_device.emplace(core, index);
            TT_FATAL(inserted, "Kernel config buffer has two shards on device {} core {}", device_id, core);
        }
        cores_ = cores_.merge(shards.cores);
        size_ = std::max(size_, shards.size);
        if (mesh_device_ != nullptr) {
            device_ids_.emplace(mesh_device_->get_view().find_device(device_id), device_id);
        }
    }
}

const KernelConfigBufferImpl::Shards* KernelConfigBufferImpl::find(ChipId device_id, const CoreCoord& core) const {
    const auto device = shards_on_device_.find(device_id);
    if (device == shards_on_device_.end()) {
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
    TT_FATAL(mesh_device_ != nullptr, "This kernel config buffer is per device: address it by device id");
    const auto it = device_ids_.find(device_coord);
    TT_FATAL(it != device_ids_.end(), "Kernel config buffer has no shard on the device at {}", device_coord);
    return it->second;
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

bool KernelConfigBufferImpl::is_allocated() const {
    return std::all_of(
        shards_.begin(), shards_.end(), [](const Shards& shards) { return shards.buffer->is_allocated(); });
}

void KernelConfigBufferImpl::deallocate() {
    for (const Shards& shards : shards_) {
        shards.buffer->impl().deallocate(*shards.buffer);
    }
}

namespace {

void check_hybrid(IDevice& device) {
    TT_FATAL(
        device.allocator_impl()->get_config().allocator_mode == AllocatorMode::HYBRID,
        "Device {}: a kernel config buffer is allocated per core, which needs the HYBRID allocator mode (set "
        "TT_METAL_ALLOCATOR_MODE_HYBRID=1 before opening the device)",
        device.id());
}

// Each core's block in `program` on its device: its kernel group's, aligned to the device's L1 alignment.
std::unordered_map<CoreCoord, uint32_t> blocks_on_cores(IDevice& device, Program& program) {
    const uint32_t alignment = device.allocator_impl()->get_alignment(BufferType::L1);
    const uint32_t index = tensix_index(program);
    const ProgramConfig& program_config = program.impl().get_program_config(index);
    std::unordered_map<CoreCoord, uint32_t> block_on_core;
    for (const auto& kernel_group : program.impl().get_kernel_groups(index)) {
        const uint32_t block = tt::align(kernel_config_footprint(*kernel_group, program_config), alignment);
        for (const CoreCoord& core : corerange_to_cores(kernel_group->core_ranges)) {
            block_on_core.emplace(core, block);
        }
    }
    return block_on_core;
}

// Allocates the shards for `program` on `device`: one per-core-allocated Buffer per block size, each core in the one of
// its block. With `lockstep_address`, one Buffer at the largest block, allocated bottom up, which must land at one
// address on every core: it does when the cores' free L1 starts at the same address.
std::vector<KernelConfigBufferImpl::Shards> allocate_shards(IDevice& device, Program& program, bool lockstep_address) {
    check_hybrid(device);
    const auto block_on_core = blocks_on_cores(device, program);
    uint32_t largest = 0;
    for (const auto& [core, block] : block_on_core) {
        largest = std::max(largest, block);
    }
    std::map<uint32_t, std::vector<CoreRange>> cores_by_size;
    for (const auto& [core, block] : block_on_core) {
        cores_by_size[lockstep_address ? largest : block].emplace_back(core);
    }

    std::vector<KernelConfigBufferImpl::Shards> shards;
    std::string sizes;
    for (const auto& [size, core_list] : cores_by_size) {
        const CoreRangeSet cores = CoreRangeSet(core_list).merge_ranges();
        const uint32_t num_cores = cores.num_cores();
        // One page per core, each core allocated on its own, so the buffer takes L1 only on its cores.
        BufferShardingArgs sharding_args(
            ShardSpecBuffer(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED);
        per_core_allocation::set_per_core_allocation(sharding_args, true);
        auto buffer = BufferImpl::create(
            &device,
            static_cast<DeviceAddr>(size) * num_cores,
            size,
            BufferType::L1,
            sharding_args,
            lockstep_address ? std::optional<bool>(true) : std::nullopt);
        if (lockstep_address) {
            // Throws if the shards landed at different addresses.
            per_core_allocation::get_shard_base_address(*buffer, cores);
        }
        sizes += fmt::format(" {} B x {} cores;", size, num_cores);
        shards.push_back({.cores = cores, .size = size, .buffer = std::move(buffer)});
    }
    log_debug(tt::LogMetal, "Kernel config buffer on device {}:{}", device.id(), sizes);
    return shards;
}

}  // namespace

void ValidateKernelConfigBuffer(Program& program, const KernelConfigBufferImpl& buffer, ChipId device_id) {
    TT_FATAL(buffer.is_allocated(), "Kernel config buffer was deallocated");
    if (!buffer.lockstep_address()) {
        check_no_tensix_semaphores(program);
    }

    // Each core holds only its kernel group's part of the block, in a shard sized for it.
    const uint32_t index = tensix_index(program);
    const ProgramConfig& program_config = program.impl().get_program_config(index);
    for (const auto& kernel_group : program.impl().get_kernel_groups(index)) {
        const uint32_t footprint = kernel_config_footprint(*kernel_group, program_config);
        for (const CoreCoord& core : corerange_to_cores(kernel_group->core_ranges)) {
            const KernelConfigBufferImpl::Shards* shards = buffer.find(device_id, core);
            TT_FATAL(
                shards != nullptr,
                "Kernel config buffer has no shard on device {} core {}, where the program runs a kernel",
                device_id,
                core);
            TT_FATAL(
                footprint <= shards->size,
                "Kernel group on {} needs a {} B kernel config block, but its kernel config buffer has {} B on device "
                "{} core {}",
                kernel_group->core_ranges,
                footprint,
                shards->size,
                device_id,
                core);
        }
    }
}

KernelConfigBuffer CreateKernelConfigBuffer(
    distributed::MeshDevice& device, distributed::MeshWorkload& workload, bool lockstep_address) {
    check_slow_dispatch(device);
    if (!lockstep_address) {
        for (auto& [_, program] : workload.get_programs()) {
            check_no_tensix_semaphores(program);
        }
    }
    // Laid out as launch lays it out, so launch reuses this. Nothing to do if the caller already compiled it.
    program_preparation::compile(device, workload);

    // Each local device gets shards for the program that runs on it.
    std::vector<KernelConfigBufferImpl::Shards> shards;
    for (auto& [device_range, program] : workload.get_programs()) {
        for (const distributed::MeshCoordinate& coord : device_range) {
            if (!device.impl().is_local(coord)) {
                continue;
            }
            auto device_shards = allocate_shards(*device.impl().get_device(coord), program, lockstep_address);
            std::move(device_shards.begin(), device_shards.end(), std::back_inserter(shards));
        }
    }
    TT_FATAL(!shards.empty(), "The workload has no Tensix kernels on this host's devices, so no kernel config block");
    return KernelConfigBuffer(std::make_shared<KernelConfigBufferImpl>(std::move(shards), lockstep_address, &device));
}

void EnqueueMeshWorkload(
    distributed::MeshCommandQueue& mesh_cq,
    distributed::MeshWorkload& workload,
    const KernelConfigBuffer& buffer,
    bool blocking) {
    distributed::MeshDevice& device = *mesh_cq.device();
    check_slow_dispatch(device);
    TT_FATAL(buffer.impl().is_allocated(), "Kernel config buffer was deallocated");
    const distributed::MeshDevice* buffer_device = buffer.impl().mesh_device();
    TT_FATAL(
        buffer_device == &device,
        "Kernel config buffer is allocated on MeshDevice {}, but the workload is enqueued on MeshDevice {}",
        buffer_device != nullptr ? std::to_string(buffer_device->id()) : std::string("(none)"),
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
