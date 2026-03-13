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

using SemaphoreSpecId = uint32_t;
using SemaphoreSpecName = std::string;

struct SemaphoreSpec { 

    // Semaphore identifier
    // A handle used to reference this Semaphore within the ProgramSpec
    std::variant<SemaphoreSpecId, SemaphoreSpecName> unique_id;  
    // (I'm considering removing either the string or uint32_t option. Both is annoying. Thoughts?)

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