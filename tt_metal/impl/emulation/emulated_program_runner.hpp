// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_metal::emule {

/// JIT-compile a kernel source file to an x86 shared library and return an
/// invocable entry point.  The returned function captures the dlopen handle
/// via shared_ptr so the .so stays loaded as long as the function exists.
///
/// @param kernel_src_path  Absolute path to the kernel .cpp source file.
/// @param compile_args     Compile-time args passed as -DKERNEL_COMPILE_TIME_ARGS=v0,v1,...
/// @param defines          Extra -D flags (key=value pairs).
/// @return                 A callable void() that invokes the kernel entry point.
std::function<void()> jit_compile_kernel(
    const std::string& kernel_src_path,
    const std::vector<uint32_t>& compile_args,
    const std::unordered_map<std::string, uint32_t>& named_compile_args = {},
    const std::map<std::string, std::string>& defines = {},
    const std::string& extra_include_flags = "");

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
