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
// BINDING SCOPE: Any kernel can bind to any semaphore in the ProgramSpec,
//   regardless of location. Any kernel instance can signal or wait on any
//   semaphore instance.
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
};

}  // namespace tt::tt_metal::experimental
