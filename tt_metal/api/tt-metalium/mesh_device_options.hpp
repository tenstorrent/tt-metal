// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/dispatch_core_common.hpp>

namespace tt::tt_metal {

// Options for opening a MeshDevice or a UnitMeshGroup: the same settings MeshDevice::create takes as positional
// parameters.
struct MeshDeviceOptions {
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE;
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE;
    uint8_t num_command_queues = 1;
    DispatchCoreConfig dispatch_core_config;
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE;
};

}  // namespace tt::tt_metal
