// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt_stl/strong_type.hpp>
#include <hostdevcommon/sem_scope.h>

namespace tt::tt_metal::experimental {

// ============================================================================
//  SemaphoreSpec API
// ============================================================================
//
// A SemaphoreSpec is a descriptor for a Tenstorrent semaphore,
// which can be used for inter-kernel instance synchronization.
//
// INSTANCING: One SRAM ("L1") cell per node in the set of target_nodes.
//
// PLACEMENT: Specified directly via target_nodes. Unlike DFBs, semaphores are
//   remote resources for kernels. Placement cannot be inferred from kernel
//   bindings.
//
// BINDING SCOPE: Any kernel can bind to any semaphore in the ProgramSpec and
//   signal it (up() takes explicit coordinates for remote targets). Consumers
//   (down()) must run on the semaphore's node -- the host rejects off-node
//   CONSUME binders at build time.
//
// ============================================================================

// A name identifying a SemaphoreSpec within a ProgramSpec.
using SemaphoreSpecName = ttsl::StrongType<std::string, struct SemaphoreSpecNameTag>;

struct SemaphoreSpec {
    // Semaphore identifier: used to reference this Semaphore within the ProgramSpec
    SemaphoreSpecName unique_id;

    // Target nodes
    Nodes target_nodes;

    //////////////////////////////////////////////////////////////////////////////
    // Advanced options (see advanced_options.hpp)
    //////////////////////////////////////////////////////////////////////////////
    SemaphoreAdvancedOptions advanced_options;

    // Physical-path INTENT for this semaphore. AUTO lets the host derive it. Forcing
    // DM_LOCAL_CACHED is validated at build time (e.g. multi-node is a FATAL); forced
    // EXTERNAL still rejects off-node CONSUME binders; LOCAL_NONATOMIC passes through
    // unvalidated. The host resolves this to a device SemScope that the kernel picks up
    // via CTAD (see noc_semaphore.h).
    SemaphoreScope scope = SemaphoreScope::AUTO;
};

}  // namespace tt::tt_metal::experimental
