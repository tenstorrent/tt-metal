// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/distributed/mesh_buffer.hpp"
#include "tt_metal/distributed/mesh_command_queue.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class IDevice;
class Tensor;

}  // namespace v0

namespace distributed {

MeshWorkload CreateMeshWorkload();

void AddProgramToMeshWorkload(MeshWorkload& mesh_workload, Program& program, const LogicalDeviceRange& device_range);

void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);

void Finish(MeshCommandQueue& mesh_cq);

}  // namespace distributed
}  // namespace tt::tt_metal
