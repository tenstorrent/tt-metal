// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <variant>

#include <tt-metalium/program.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace tt {

namespace tt_metal {

class IDevice;

namespace experimental {

// clang-format off
/**
 * Create a kernel from pre-compiled binaries for environments without source files.
 *
 * WORKFLOW:
 * 1. Run your program with normal JIT compilation (calling CreateKernel()) to populate cache (~/.cache/tt-metal-cache/)
 * 2. Copy kernel binaries from cache to your deployment location, preserving directory structure
 * 3. Call SetKernelBinaryPathPrefix() to set the deployment location
 * 4. Use CreateKernelFromBinary() instead of CreateKernel()
 *
 * CACHE STRUCTURE:
 * The JIT cache generates paths like:
 * ~/.cache/tt-metal-cache/<git_hash>/<build_key>/kernels/<kernel_name>/<kernel_hash>/<processor>/<processor>.elf
 *
 * Where:
 * - git_hash: Current git commit (10 chars, e.g., "3493929f10")
 * - build_key: Build configuration hash (architecture, flags, defines)
 * - kernel_name: Name from source file without .cpp (e.g., "simple_add")
 * - kernel_hash: Compilation configuration hash (includes compile-time args, defines, etc.)
 * - processor: Processor type (brisc, ncrisc, trisc0, trisc1, trisc2, erisc)
 *
 * IMPORTANT: The same kernel can have multiple kernel_hash values if used with different compile-time
 * arguments. You must copy ALL hash directories for a kernel.
 *
 * EXAMPLE:
 * ```cpp
 * // Set the binary path (call once per device)
 * SetKernelBinaryPathPrefix(device, "/path/to/binaries");
 *
 * // Compute hash of original path (can be done once and hardcoded)
 * auto hash = ComputeKernelOriginalPathHash("path/to/kernel.cpp");
 *
 * // Create kernel from binary (replaces CreateKernel)
 * auto kernel = CreateKernelFromBinary(
 *     program, "kernel_name", core_spec, config, hash
 * );
 * ```
 *
 * | Argument               | Description                                                                                                    | Type                                                                     | Valid Range | Required |
 * |------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|-------------|----------|
 * | program                | The program to which this kernel will be added to                                                              | Program &                                                                |             | Yes      |
 * | kernel_name            | Name of the pre-compiled kernel binary (name of source file without .cpp)                                      | const std::string &                                                      |             | Yes      |
 * | core_spec              | Either a single logical core, a range of logical cores or a set of logical core ranges for kernel placement    | const std::variant<CoreCoord, CoreRange, CoreRangeSet> &                 |             | Yes      |
 * | config                 | Config for data movement, compute, or ethernet kernel (must match the config used during JIT compilation)      | const std::variant<DataMovementConfig,ComputeConfig,EthernetConfig> &    |             | Yes      |
 * | original_path_or_hash  | Original kernel source path (string) or pre-computed hash (from ComputeKernelOriginalPathHash())               | const std::variant<std::string, size_t> &                                |             | Yes      |
 */
// clang-format on
KernelHandle CreateKernelFromBinary(
    Program& program,
    const std::string& kernel_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::variant<std::string, size_t>& original_path_or_hash);

// clang-format off
/**
 * Set the root directory containing pre-compiled kernel binaries.
 *
 * The directory structure must be:
 * <binary_path_prefix>/kernels/<kernel_name>/<kernel_hash>/<processor>/<processor>.elf
 *
 * This directory structure matches the cache structure (excluding git_hash and build_key components).
 * You can copy it from: ~/.cache/tt-metal-cache/<git_hash>/<build_key>/
 *
 * Must be called before CreateKernelFromBinary().
 *
 * EXAMPLE:
 * ```cpp
 * // For architecture-specific binaries:
 * std::string binary_path;
 * if (device->arch() == tt::ARCH::WORMHOLE_B0) {
 *     binary_path = "deployment/wormhole/kernels";
 * } else if (device->arch() == tt::ARCH::BLACKHOLE) {
 *     binary_path = "deployment/blackhole/kernels";
 * }
 * experimental::SetKernelBinaryPathPrefix(device, binary_path);
 * ```
 *
 * Return value: void
 *
 * | Argument           | Description                                   | Type                | Valid Range | Required |
 * |--------------------|-----------------------------------------------|---------------------|-------------|----------|
 * | device             | The device to set the binary path prefix for | IDevice*             |             | Yes      |
 * | binary_path_prefix | Root path to directory containing kernels/   | const std::string & |             | Yes      |
 */
// clang-format on
void SetKernelBinaryPathPrefix(IDevice* device, const std::string& binary_path_prefix);

// clang-format off
/**
 * Compute a hash of the original kernel source path for use with CreateKernelFromBinary.
 *
 * The kernel compilation hash includes a hash of the original source file path. This function
 * allows you to compute that hash once (during development) and use it in production code
 * instead of passing the path as a string.
 *
 * RECOMMENDED WORKFLOW:
 * 1. During development, call this function to get the hash for each kernel
 * 2. Store these hashes as constants in your production code
 * 3. In production, pass the hash to CreateKernelFromBinary instead of the path string
 *
 * EXAMPLE:
 * ```cpp
 * // Development: compute hash
 * auto hash = ComputeKernelOriginalPathHash("tt_metal/kernels/compute/simple_add.cpp");
 * // Output: 7112760208363739310
 *
 * // Production: use hardcoded hash
 * constexpr size_t SIMPLE_ADD_HASH = 7112760208363739310ULL;
 * CreateKernelFromBinary(program, "simple_add", core_spec, config, SIMPLE_ADD_HASH);
 * ```
 *
 * Return value: size_t - Hash of the path that can be passed to CreateKernelFromBinary
 *
 * | Argument      | Description                                            | Type                | Valid Range | Required |
 * |---------------|--------------------------------------------------------|---------------------|-------------|----------|
 * | original_path | The original path of the kernel source (the .cpp file) | const std::string & |             | Yes      |
 */
// clang-format on
size_t ComputeKernelOriginalPathHash(const std::string& original_path);

}  // namespace experimental

}  // namespace tt_metal

}  // namespace tt
