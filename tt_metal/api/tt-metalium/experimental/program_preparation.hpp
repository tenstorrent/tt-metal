// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/mesh_workload.hpp>

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental::program_preparation {

/// Program-memory use established by non-dispatch workload preparation, reported for diagnostics and measurement.
/// prepare() itself rejects a workload that does not fit, so callers need not compare these sizes to a limit.
struct ProgramCapacity {
    uint32_t max_program_config_size_bytes = 0;
    uint32_t max_kernel_binary_size_bytes = 0;
};

/// Compiles kernels, finalizes program offsets and runtime arguments, and validates capacity without dispatching.
/// Throws if `mesh_device` has no local devices (an enqueue would do nothing), if `workload` has no programs or was
/// finalized for another MeshDevice, or if compilation fails, including when its program configuration does not fit
/// the kernel-configuration buffer.
ProgramCapacity prepare(distributed::MeshWorkload& workload, distributed::MeshDevice& mesh_device);

}  // namespace tt::tt_metal::experimental::program_preparation
