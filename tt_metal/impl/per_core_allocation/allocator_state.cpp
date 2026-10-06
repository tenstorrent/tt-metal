// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/allocator_state.hpp>

#include <algorithm>
#include <tt_stl/assert.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "impl/allocator/allocator.hpp"

namespace tt::tt_metal::experimental::per_core_allocation {

namespace {

// L1 on a chip is tracked by the chip's own allocator and by the MeshDevice allocator, which holds
// mesh-level persistent L1 (e.g. PrefetcherPipe) and buffers created directly on the MeshDevice.
std::vector<const AllocatorImpl*> hybrid_allocators(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord) {
    TT_FATAL(mesh_device.impl().is_local(device_coord), "get_l1_occupied_ranges: device {} is not local", device_coord);
    const AllocatorImpl* device_allocator = mesh_device.impl().get_device(device_coord)->allocator_impl().get();
    const AllocatorImpl* mesh_allocator = mesh_device.allocator_impl().get();
    TT_FATAL(
        device_allocator->get_config().allocator_mode == AllocatorMode::HYBRID,
        "get_l1_occupied_ranges requires AllocatorMode::HYBRID when opening the device");
    if (mesh_allocator == device_allocator) {
        return {device_allocator};
    }
    return {device_allocator, mesh_allocator};
}

AddressRanges occupied_ranges(const std::vector<const AllocatorImpl*>& allocators, const CoreCoord& core) {
    using AllocatorID = BankManager::AllocatorDependencies::AllocatorID;
    TT_FATAL(
        allocators.front()->find_bank_ids(BufferType::L1, core) != nullptr,
        "get_l1_occupied_ranges: core {} has no L1 bank",
        core.str());

    AddressRanges ranges;
    auto append = [&ranges](const AddressRanges& more) { ranges.insert(ranges.end(), more.begin(), more.end()); };
    for (const AllocatorImpl* allocator : allocators) {
        append(allocator->persistent_l1().occupied_ranges(core));
        const auto* bank_ids = allocator->find_bank_ids(BufferType::L1, core);
        if (bank_ids == nullptr) {
            continue;
        }
        append(allocator->get_l1_allocated_ranges(AllocatorID{0}));
        // Same bank the per-core allocation path uses for this core.
        append(allocator->get_l1_allocated_ranges(AllocatorID{bank_ids->front() + 1}));
    }
    std::sort(ranges.begin(), ranges.end());

    AddressRanges merged;
    for (const auto& [start, end] : ranges) {
        if (!merged.empty() && start <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, end);
        } else {
            merged.emplace_back(start, end);
        }
    }
    return merged;
}

}  // namespace

AddressRanges get_l1_occupied_ranges(
    const distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& core) {
    return occupied_ranges(hybrid_allocators(mesh_device, device_coord), core);
}

AddressRanges get_l1_free_ranges(
    const distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& device_coord,
    const CoreCoord& core) {
    const auto allocators = hybrid_allocators(mesh_device, device_coord);
    const AllocatorImpl* device_allocator = allocators.front();
    const auto* bank_ids = device_allocator->find_bank_ids(BufferType::L1, core);
    TT_FATAL(bank_ids != nullptr && !bank_ids->empty(), "get_l1_free_ranges: core {} has no L1 bank", core.str());

    AddressRanges external_occupied;
    auto append = [&external_occupied](const AddressRanges& ranges) {
        external_occupied.insert(external_occupied.end(), ranges.begin(), ranges.end());
    };
    for (const AllocatorImpl* allocator : allocators) {
        append(allocator->persistent_l1().occupied_ranges(core));
        if (allocator != device_allocator) {
            append(allocator->get_l1_allocated_ranges(BankManager::AllocatorDependencies::AllocatorID{0}));
            if (const auto* other_bank_ids = allocator->find_bank_ids(BufferType::L1, core)) {
                append(allocator->get_l1_allocated_ranges(
                    BankManager::AllocatorDependencies::AllocatorID{other_bank_ids->front() + 1}));
            }
        }
    }
    return device_allocator->get_l1_available_ranges(
        BankManager::AllocatorDependencies::AllocatorID{bank_ids->front() + 1}, external_occupied);
}

std::unordered_map<CoreCoord, AddressRanges> get_l1_occupied_ranges(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& device_coord) {
    const auto allocators = hybrid_allocators(mesh_device, device_coord);
    const AllocatorImpl& device_allocator = *allocators.front();
    std::unordered_map<CoreCoord, AddressRanges> ranges_by_core;
    for (uint32_t bank_id = 0; bank_id < device_allocator.get_num_banks(BufferType::L1); bank_id++) {
        const CoreCoord core = device_allocator.get_logical_core_from_bank_id(bank_id);
        if (!ranges_by_core.contains(core)) {
            ranges_by_core.emplace(core, occupied_ranges(allocators, core));
        }
    }
    return ranges_by_core;
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
