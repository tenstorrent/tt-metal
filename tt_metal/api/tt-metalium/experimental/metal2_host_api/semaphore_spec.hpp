// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// A name identifying a SemaphoreSpec within a ProgramSpec.
//
// CONVENTION: define names as `constexpr const char*` constants, e.g.:
//   constexpr const char* DONE_FLAG = "done_flag";
//   SemaphoreSpec{.unique_id = DONE_FLAG, ...};
// Reusing a single constant helps catch typos and errors at compile time.
using SemaphoreSpecName = std::string;

// A SemaphoreSpec is a descriptor for a Tenstorrent semaphore,
// which can be used for inter-kernel instance synchronization.
//
// Instancing: One SRAM ("L1") cell per node in the set of target_nodes.
//
// Placement: Specified directly via target_nodes. Unlike DFBs, semaphores are remote
// resources for kernels. Placement cannot be inferred from kernel bindings.
//
// Binding scope: Any kernel can bind to any semaphore in the ProgramSpec, regardless of
// location. Any kernel instance can signal or wait on any semaphore instance.
//
struct SemaphoreSpec {
    // Semaphore identifier: used to reference this Semaphore within the ProgramSpec
    SemaphoreSpecName unique_id;

    // Target nodes
    using Nodes = std::variant<NodeCoord, NodeRange, NodeRangeSet>;
    Nodes target_nodes;

    //////////////////////////////
    // Advanced options
    //////////////////////////////

    // Initial value
    // NOTE: Setting a non-zero initial value is not supported on Gen2 architectures.
    // NOTE: Runtime wants to deprecate this feature for ALL architectures.
    //       When remote DFB becomes available, non-zero initial values will be removed.
    uint32_t initial_value = 0;

    // Backing memory
    // NOTE: Register-backed semaphores are a Gen2-only hardware feature.
    //       They are not yet supported; TBD whether we will ever support them.
    enum class SemaphoreMemoryType { L1, Register };
    SemaphoreMemoryType memory_type = SemaphoreMemoryType::L1;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
