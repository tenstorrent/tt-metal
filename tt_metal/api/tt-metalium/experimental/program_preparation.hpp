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

/// Compiles the kernels of `workload` and lays out its programs for `mesh_device` as its enqueue would, without
/// dispatching it. Unlike `prepare`, does not check the kernel-configuration blocks against the KERNEL_CONFIG window in
/// slow dispatch: launch checks them against the window, or against the kernel config buffer it runs from
/// (experimental::CreateKernelConfigBuffer). Fast dispatch launches only from the window, so there it is `prepare`.
/// Throws if `mesh_device` has no local devices, if `workload` has no programs, or if compilation fails.
void compile(distributed::MeshDevice& mesh_device, distributed::MeshWorkload& workload);

}  // namespace tt::tt_metal::experimental::program_preparation
