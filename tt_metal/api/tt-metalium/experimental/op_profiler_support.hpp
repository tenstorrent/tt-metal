// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/kernel_types.hpp>

namespace tt::tt_metal {

class Program;
namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// Only used in op_profiler, we might want to expose this via a tooling interface instead of through here.
// Collects the meta data of kernels in a program, and the metadata of the binaries within the kernel for mesh_device.
std::vector<detail::KernelMeta> collect_kernel_meta(const Program& program, distributed::MeshDevice& mesh_device);

}  // namespace experimental

}  // namespace tt::tt_metal
