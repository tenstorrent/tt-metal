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

// Inspector-only annotation for correlating higher-level execution with MeshWorkload runs.
void EmitMeshWorkloadAnnotation(
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation_name,
    std::string_view operation_parameters);

// Inspector-only runtime id for correlating workload enqueues/runs across tools.
void EmitMeshWorkloadRuntimeId(tt::tt_metal::distributed::MeshWorkload& workload, uint64_t runtime_id);

}  // namespace tt::tt_metal::experimental::inspector
