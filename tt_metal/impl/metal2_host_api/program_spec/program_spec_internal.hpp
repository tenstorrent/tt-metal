// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

namespace tt::tt_metal {

// Builds a Program from a ProgramSpec without compiling it. MakeProgramFromSpec is this followed by
// compile_and_allocate. Internal: for callers (tests) that must inspect or drive compile themselves.
Program BuildProgramFromSpec(
    distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation = false);

}  // namespace tt::tt_metal
