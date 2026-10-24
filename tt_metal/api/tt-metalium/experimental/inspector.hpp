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
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::inspector {

// Whether the Inspector is active for the context `mesh_device` belongs to.
bool IsEnabled(const tt::tt_metal::distributed::MeshDevice& mesh_device);

// Whether tensor spec capture on op dispatch is enabled for the context `mesh_device` belongs to.
bool ShouldCaptureTensorSpecs(const tt::tt_metal::distributed::MeshDevice& mesh_device);

// Returns the current mesh trace id for `mesh_device` if fast dispatch is enabled; otherwise nullopt.
// Fetching the trace id is only supported on fast dispatch and would throw on slow dispatch.
std::optional<tt::tt_metal::distributed::MeshTraceId> GetCurrentMeshTraceId(
    const tt::tt_metal::distributed::MeshDevice& mesh_device);

// Emit a debug entry for a mesh workload execution on `mesh_device`, capturing the operation name and tensor specs.
void EmitMeshWorkloadDebugEntry(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    uint64_t runtime_id,
    std::string_view operation_name,
    std::vector<TensorSpec> tensor_specs,
    std::optional<tt::tt_metal::distributed::MeshTraceId> trace_id = std::nullopt);

}  // namespace tt::tt_metal::experimental::inspector
