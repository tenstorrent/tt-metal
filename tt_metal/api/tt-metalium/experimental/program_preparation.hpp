// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental::program_preparation {

/// Compiles the kernels of `workload` and finalizes its program offsets and runtime arguments for `mesh_device`,
/// without dispatching it.
/// Throws if `mesh_device` has no local devices (an enqueue would do nothing), if `workload` has no programs or was
/// finalized for another MeshDevice, or if compilation fails, including when a program's configuration does not fit
/// the kernel-configuration buffer.
void prepare(distributed::MeshDevice& mesh_device, distributed::MeshWorkload& workload);

}  // namespace tt::tt_metal::experimental::program_preparation
