// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

namespace tt::tt_metal {

// Launch `program` on every device in `mesh_device` via EnqueueMeshWorkload, blocking until completion.
//
// Exception: a service workload (a program targeting claimed service cores) is routed by
// EnqueueMeshWorkload to slow dispatch, which rings GO and returns without waiting - `blocking` is not
// consulted on that path. Callers that launch onto claimed service cores must synchronize themselves.
//
// @return MeshWorkload that takes ownership of the program.
inline distributed::MeshWorkload LaunchProgram(distributed::MeshDevice& mesh_device, Program&& program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(0), workload, /*blocking=*/true);
    return workload;
}

}  // namespace tt::tt_metal
