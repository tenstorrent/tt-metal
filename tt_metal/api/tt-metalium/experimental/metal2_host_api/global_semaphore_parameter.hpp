// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/buffer_types.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  GlobalSemaphoreParameter API
// ============================================================================
//
// A GlobalSemaphoreParameter declares that a Program uses a user-managed GlobalSemaphore.
//
// A GlobalSemaphore has the same address on every node it spans, on every device of its
// MeshDevice. Thus a kernel on one device can signal the GlobalSemaphore of a peer device
// (for example, through the fabric).
//
// The GlobalSemaphore object is supplied at Program execution time, via
// AdvancedProgramRunArgs::global_semaphore_args. A different GlobalSemaphore may be supplied
// for each execution.
//
// ============================================================================

// A name identifying a GlobalSemaphoreParameter within a ProgramSpec.
using GlobalSemaphoreParamName = ttsl::StrongType<std::string, struct GlobalSemaphoreParamNameTag>;

struct GlobalSemaphoreParameter {
    // Identifier: used to reference this GlobalSemaphore within the ProgramSpec
    GlobalSemaphoreParamName unique_id;

    // The memory that the supplied GlobalSemaphore is allocated in.
    // Invariant: L1 or L1_SMALL.
    BufferType memory_type = BufferType::L1;
};

}  // namespace tt::tt_metal::experimental
