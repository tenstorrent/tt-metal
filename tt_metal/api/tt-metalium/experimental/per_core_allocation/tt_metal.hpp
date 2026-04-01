// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt-metalium/experimental/per_core_allocation/allocator_mode.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental::per_core_allocation {

std::map<ChipId, IDevice*> CreateDevices(
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs = 1,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
    const std::vector<uint32_t>& l1_bank_remap = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE,
    bool init_profiler = true,
    bool initialize_fabric_and_dispatch_fw = true,
    AllocatorMode allocator_mode = AllocatorMode::HYBRID);

}  // namespace tt::tt_metal::experimental::per_core_allocation
