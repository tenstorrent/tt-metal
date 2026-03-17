// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

using SemaphoreSpecName = std::string;

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
    // Runtime wants to deprecate this feature for ALL architectures
    uint32_t initial_value = 0;

    // Backing memory
    // NOTE: Register-backed semaphores are only supported on Gen2 architectures.
    enum class SemaphoreMemoryType { L1, Register };
    SemaphoreMemoryType memory_type = SemaphoreMemoryType::L1;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
