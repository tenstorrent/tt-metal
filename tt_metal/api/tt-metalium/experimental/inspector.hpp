// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::experimental::inspector {

// Inspector feature flag
bool IsEnabled();

// Whether tensor spec capture is enabled on op dispatch (checks rtoptions).
bool CaptureTensorSpecs();

// Emit a debug entry for a mesh workload execution, capturing the operation name and tensor specs.
void EmitMeshWorkloadDebugEntry(
    tt::tt_metal::distributed::MeshWorkload& workload,
    uint64_t runtime_id,
    std::string_view operation_name,
    std::vector<TensorSpec> tensor_specs);

}  // namespace tt::tt_metal::experimental::inspector
