// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::internal {

// INTERNAL API: Resolve the sub-devices occupied by a workload using the same implementation as
// mesh dispatch. This exists for TTNN graph-report tooling and is not part of the stable Metalium API.
std::unordered_set<SubDeviceId> get_mesh_workload_sub_device_ids(
    distributed::MeshWorkload& mesh_workload, distributed::MeshDevice* mesh_device);

}  // namespace tt::tt_metal::internal
