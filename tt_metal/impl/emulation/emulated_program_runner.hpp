// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::emule {

/// Execute all kernels in a Program on the emulated device.
///
/// For each kernel in the program:
///   1. Resolve source path from KernelSource
///   2. JIT compile to x86 shared library (cached by kernel hash)
///   3. For each core in the kernel's CoreRangeSet, spawn a thread that:
///      - Sets thread-local context (__rt_args, __core_coord, __device_ptr)
///      - Invokes the JIT'd entry point
///   4. Join all threads
///
/// Memory I/O from kernels goes through the SWEmuleChip's backing store
/// via UMD Cluster::write_core / Cluster::read_core.
///
/// `post_setup_barrier`, if set, is invoked after this device finishes writing its per-core CB/semaphore
/// initial values to L1 but BEFORE it launches kernels. Under concurrent multi-device dispatch it must
/// block until every participating device has reached the same point, so a peer's fabric teleport (e.g.
/// a barrier / out-ready atomic-inc) can never land before this device has initialized that semaphore.
void execute_program_emulated(
    IDevice* device, Program& program, const std::function<void()>& post_setup_barrier = {});

}  // namespace tt::tt_metal::emule
