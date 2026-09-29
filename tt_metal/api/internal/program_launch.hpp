// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

namespace tt::tt_metal {

class IDevice;
class Program;

namespace internal {

/**
 * @warning INTERNAL. Everything declared in this header lives under
 * @c api/internal: it exists to serve tt-metal's own runtime and tooling, is
 * not part of the supported user-facing API, and may change or be removed
 * without a deprecation period.
 *
 * These are the direct, command-queue-free program launch entry points. User
 * code dispatches programs with @c distributed::EnqueueMeshWorkload, which is
 * built on top of these. They remain reachable here because tt-metal's own
 * slow-dispatch path (@c SDMeshCommandQueue) and the device profiler must
 * launch a program outside any command queue.
 */

// Launches all kernels on cores specified with kernels in the program.
// All kernels on a given Tensix core must be launched.
void LaunchProgram(
    IDevice* device, Program& program, bool wait_until_cores_done = true, bool force_slow_dispatch = false);
void LaunchProgram(
    IDevice* device,
    const std::shared_ptr<Program>& program,
    bool wait_until_cores_done = true,
    bool force_slow_dispatch = false);

/**
 *  Compiles all kernels within the program, and generates binaries that are written to
 * `<tt-metal-cache directory>/<build_key>/kernels/<kernel name>/<kernel hash>`
 *
 *  The build key component accounts for device architecture as binaries are not compatible across architectures.
 *  To speed up compilation there is a kernel compilation cache that skips over generating binaries for the previously
 * compiled kernels. Kernel uniqueness is determined by the kernel hash which is computed based on compile time args,
 * defines, and kernel type specific attributes such as NOC for data movement kernels and math fidelity for compute
 * kernels.
 *  On cache hits the kernel is not recompiled if the output binary directory exists, otherwise the kernel is compiled.
 *  This cache is static and is enabled for the duration of the running process.
 *  Across runs, previously compiled kernels are recompiled if the source code or dependencies have changed.
 *
 *  Programs are compiled automatically by the runtime infrastructure; call this only when a program must be built
 *  ahead of the launch that would otherwise do it.
 *
 *  Return value: void
 *
 * | Argument                  | Description                                                      | Type      | Valid
 * Range                                        | Required |
 * |---------------------------|------------------------------------------------------------------|-----------|----------------------------------------------------|----------|
 * | device                    | Which device the program is compiled for                         | IDevice*  | Must be
 * initialized via tt_metal::InitializeDevice | Yes      | | program                   | The program to compile |
 * Program & |                                                    | Yes      | | force_slow_dispatch        | Set when
 * a user wants to compile a program with Slow Dispatch Force Enabled (advanced feature, currently used internally to
 * launch Fast Dispatch Firmware and in the Device Performance Profiler)           | bool      | | No |
 */
void CompileProgram(IDevice* device, Program& program, bool force_slow_dispatch = false);

}  // namespace internal

}  // namespace tt::tt_metal
