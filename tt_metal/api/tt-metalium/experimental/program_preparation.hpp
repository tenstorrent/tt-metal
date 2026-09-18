// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/mesh_workload.hpp>

namespace tt::tt_metal::experimental::program_preparation {

/// Program-memory use established by non-dispatch workload preparation.
struct ProgramCapacity {
    uint32_t max_program_config_size_bytes = 0;
    uint32_t max_kernel_binary_size_bytes = 0;
};

/// Compiles kernels, finalizes program offsets and runtime arguments, and validates capacity without dispatching.
ProgramCapacity prepare(distributed::MeshWorkload& workload, distributed::MeshDevice* mesh_device);

}  // namespace tt::tt_metal::experimental::program_preparation
