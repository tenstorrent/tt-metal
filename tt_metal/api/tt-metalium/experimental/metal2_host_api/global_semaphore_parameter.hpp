// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  GlobalSemaphoreParameter API
// ============================================================================
//
// A GlobalSemaphoreParameter declares that a Program uses a GlobalSemaphore: a
// user-managed semaphore whose lifetime is not bound to the Program, so it can
// carry state across Programs and across devices.
//
// Like a TensorParameter, the ProgramSpec only names the parameter; the actual
// GlobalSemaphore object is supplied at execution time via
// ProgramRunArgs::global_semaphore_args. A kernel reaches it through a
// KernelSpec::SemaphoreBinding whose semaphore_spec_name holds this parameter's
// GlobalSemaphoreParamName, and constructs it in device code as
//   GlobalSemaphore(sem::<accessor_name>)
//
// The supplied GlobalSemaphore must be created with
// experimental::CreatePersistentGlobalSemaphore (so it has the same L1 address on
// every core and device), on the Program's MeshDevice, and on every node of every
// kernel that binds it.
//
// Binding a different GlobalSemaphore to the same parameter changes only the
// per-launch address table, never the kernel binary.
//
// ============================================================================

// A name identifying a GlobalSemaphoreParameter within a ProgramSpec.
using GlobalSemaphoreParamName = ttsl::StrongType<std::string, struct GlobalSemaphoreParamNameTag>;

struct GlobalSemaphoreParameter {
    // GlobalSemaphore identifier: used to reference this GlobalSemaphore within the ProgramSpec
    GlobalSemaphoreParamName unique_id;
};

}  // namespace tt::tt_metal::experimental
