// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/per_core_allocation/mesh_device.hpp>
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental::per_core_allocation {

std::shared_ptr<distributed::MeshDevice> create_mesh_device(
    const distributed::MeshDeviceConfig& config,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    ttsl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size,
    AllocatorMode allocator_mode) {
    return distributed::MeshDeviceImpl::create(
        config,
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config,
        l1_bank_remap,
        worker_l1_size,
        allocator_mode);
}

std::shared_ptr<distributed::MeshDevice> create_unit_mesh(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    ttsl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size,
    AllocatorMode allocator_mode) {
    return distributed::MeshDeviceImpl::create_unit_mesh(
        device_id,
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config,
        l1_bank_remap,
        worker_l1_size,
        allocator_mode);
}

std::map<int, std::shared_ptr<distributed::MeshDevice>> create_unit_meshes(
    const std::vector<int>& device_ids,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    ttsl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size,
    AllocatorMode allocator_mode) {
    return distributed::MeshDeviceImpl::create_unit_meshes(
        device_ids,
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config,
        l1_bank_remap,
        worker_l1_size,
        allocator_mode);
}

}  // namespace tt::tt_metal::experimental::per_core_allocation
