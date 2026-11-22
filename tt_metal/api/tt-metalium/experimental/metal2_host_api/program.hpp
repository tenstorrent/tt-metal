// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include <unordered_map>

namespace tt::tt_metal::experimental {

// Program(mesh_device, spec), except that with defer_compile it skips the trailing compile + allocation step and
// returns a built but uncompiled Program. The caller compiles it before any dispatch.
Program MakeProgramFromSpec(distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool defer_compile);

// MeshWorkload(mesh_device, program_specs), except that with defer_compile it skips the trailing
// MeshWorkloadImpl::compile step and returns an UNCOMPILED, UNFINALIZED MeshWorkload. This exists for callers that
// only want the Programs materialized so their kernels can be JIT-compiled later, in bulk and in parallel (see
// ttnn::up_front_compile). Setting run args on the result is still legal: set_program_run_args is order-agnostic with
// respect to allocation.
//
// PRE-CONDITION: a deferred MeshWorkload MUST NOT be enqueued or otherwise treated as dispatch-ready until it has
// been compiled. The intended consumers are compile-only paths that discard the workload afterwards.
distributed::MeshWorkload MakeMeshWorkloadFromSpecs(
    distributed::MeshDevice& mesh_device,
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec>& program_specs,
    bool defer_compile);

}  // namespace tt::tt_metal::experimental
