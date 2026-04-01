// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include <tt_stl/span.hpp>
#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::per_core_allocation {

using distributed::MeshDevice;

std::shared_ptr<MeshDevice> create_mesh_device(
    const distributed::MeshDeviceConfig& config,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    size_t num_command_queues = 1,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    ttsl::Span<const std::uint32_t> l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    AllocatorMode allocator_mode = AllocatorMode::HYBRID);

std::shared_ptr<MeshDevice> create_unit_mesh(
    int device_id,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    size_t num_command_queues = 1,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    ttsl::Span<const std::uint32_t> l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    AllocatorMode allocator_mode = AllocatorMode::HYBRID);

std::map<int, std::shared_ptr<MeshDevice>> create_unit_meshes(
    const std::vector<int>& device_ids,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    size_t num_command_queues = 1,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    ttsl::Span<const std::uint32_t> l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    AllocatorMode allocator_mode = AllocatorMode::HYBRID);

}  // namespace tt::tt_metal::experimental::per_core_allocation
