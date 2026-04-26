// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::experimental::inspector {

// Inspector feature flag
bool IsEnabled();

// Whether tensor spec capture is enabled on op dispatch (checks rtoptions).
bool ShouldCaptureTensorSpecs();

// Emit a debug entry for a mesh workload execution, capturing the operation name and tensor specs.
void EmitMeshWorkloadDebugEntry(
    tt::tt_metal::distributed::MeshWorkload& workload,
    uint64_t runtime_id,
    std::string_view operation_name,
    std::vector<TensorSpec> tensor_specs,
    std::optional<tt::tt_metal::distributed::MeshTraceId> trace_id = std::nullopt);

// Drops the per-trace runtime-entry bucket. Called at trace release time.
void ReleaseTraceDebugEntries(tt::tt_metal::distributed::MeshTraceId trace_id);

}  // namespace tt::tt_metal::experimental::inspector
