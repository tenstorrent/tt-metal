// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <vector>

#include "impl/context/context_id.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {

class IDevice;
class DispatchCoreConfig;

namespace detail {

// CreateDevices but with ContextID already known
std::map<ChipId, IDevice*> CreateDevices(
    ContextId context_id,
    const std::vector<ChipId>& device_ids,
    uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    const std::vector<uint32_t>& l1_bank_remap,
    size_t worker_l1_size,
    bool init_profiler,
    bool ignored,
    bool initialize_fabric_and_dispatch_fw);

}  // namespace detail
}  // namespace tt::tt_metal
