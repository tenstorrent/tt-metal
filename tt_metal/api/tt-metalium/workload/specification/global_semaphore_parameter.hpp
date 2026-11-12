// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal {

// ============================================================================
//  GlobalSemaphoreParameter API
// ============================================================================
//
// A GlobalSemaphoreParameter declares that a Program uses a GlobalSemaphore
// (tt-metalium/global_semaphore.hpp). The GlobalSemaphore object is supplied
// at Program execution time, via ProgramRunArgs::global_semaphore_args.
//
// Like a MeshTensor, a GlobalSemaphore is a user-managed resource: its lifetime
// is not bound to the Program, and its value is kept across Program executions.
// It has the same L1 address on each of its cores, on every chip of its MeshDevice.
//
// KERNEL ACCESS: A data-movement kernel binds the parameter via
//   KernelSpec::global_semaphore_bindings, and constructs the device object from
//   the binding token (api/dataflow/global_semaphore.h):
//     GlobalSemaphore sem(global_sem::<accessor_name>);
//   The semaphore's address is passed as a common runtime arg, so binding a
//   different GlobalSemaphore does not recompile the kernel.
//
// ============================================================================

// A name identifying a GlobalSemaphoreParameter within a ProgramSpec.
using GlobalSemaphoreParamName = ttsl::StrongType<std::string, struct GlobalSemaphoreParamNameTag>;

struct GlobalSemaphoreParameter {
    // GlobalSemaphore identifier: used to reference this GlobalSemaphore within the ProgramSpec
    GlobalSemaphoreParamName unique_id;
};

}  // namespace tt::tt_metal
