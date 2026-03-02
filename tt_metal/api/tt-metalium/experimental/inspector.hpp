// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::experimental::inspector {

// Inspector feature flag
bool IsEnabled();

// Unified debug entry: emits operation name, parameters, and runtime ID in a single call.
void EmitMeshWorkloadDebugEntry(
    tt::tt_metal::distributed::MeshWorkload& workload,
    uint64_t runtime_id,
    std::string_view operation_name,
    std::string_view operation_parameters);

}  // namespace tt::tt_metal::experimental::inspector
