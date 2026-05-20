// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
void execute_program_emulated(IDevice* device, Program& program);

}  // namespace tt::tt_metal::emule
