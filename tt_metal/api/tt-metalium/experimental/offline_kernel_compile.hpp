// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>
#include <variant>
#include <cstddef>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal {
class Program;
}

namespace tt::tt_metal::experimental {

/**
 * Configuration for loading precompiled (offline-compiled) kernels.
 *
 * Use this when you have built kernel binaries ahead of time (e.g. via an
 * offline compile step or CI) and want to load them instead of JIT-compiling
 * at runtime. Set precompiled_dir to the directory containing the built
 * binaries and choose a FallbackPolicy for when a matching binary is missing.
 */
struct PrecompiledKernelConfig {
    /// Behavior when no precompiled binary is found for the requested kernel/hash.
    enum class FallbackPolicy {
        /// Compile the kernel at runtime (JIT) and use it; no error.
        JitCompile,
        /// Throw PrecompiledKernelNotFoundError and do not fall back to JIT.
        Error,
    };

    /// Directory path to search for precompiled kernel binaries.
    std::string precompiled_dir;
    /// What to do when a binary is not found. Default: Error.
    FallbackPolicy fallback_policy = FallbackPolicy::Error;
};

/**
 * Exception thrown when a precompiled kernel binary is required but not found,
 * and PrecompiledKernelConfig::fallback_policy is FallbackPolicy::Error.
 *
 * Access kernel_name(), compile_hash(), precompiled_dir(), and fallback_policy()
 * for diagnostics or to retry with a different path or policy.
 */
class PrecompiledKernelNotFoundError : public std::runtime_error {
public:
    PrecompiledKernelNotFoundError(
        std::string kernel_name,
        size_t compile_hash,
        std::string precompiled_dir,
        PrecompiledKernelConfig::FallbackPolicy fallback_policy);

    const std::string& kernel_name() const noexcept { return kernel_name_; }
    size_t compile_hash() const noexcept { return compile_hash_; }
    const std::string& precompiled_dir() const noexcept { return precompiled_dir_; }
    PrecompiledKernelConfig::FallbackPolicy fallback_policy() const noexcept { return fallback_policy_; }

private:
    std::string kernel_name_;
    size_t compile_hash_;
    std::string precompiled_dir_;
    PrecompiledKernelConfig::FallbackPolicy fallback_policy_;
};

/**
 * Create a kernel in the program by loading a precompiled binary from disk.
 *
 * Looks up the binary under precompiled_config.precompiled_dir using the
 * kernel source file name and a hash of the compile configuration. If found,
 * the binary is loaded; otherwise behavior is determined by
 * precompiled_config.fallback_policy (JIT compile or throw
 * PrecompiledKernelNotFoundError).
 *
 * @param program        Program to add the kernel to.
 * @param file_name      Source file name of the kernel (used to locate the binary).
 * @param core_spec      Core placement: single CoreCoord, CoreRange, or CoreRangeSet.
 * @param config         Kernel config: DataMovementConfig or ComputeConfig.
 * @param precompiled_config  Where to find binaries and how to handle missing ones.
 * @return KernelHandle for the created kernel.
 * @throws PrecompiledKernelNotFoundError when binary is missing and policy is Error.
 */
KernelHandle CreateKernelFromPrecompiled(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const PrecompiledKernelConfig& precompiled_config);

}  // namespace tt::tt_metal::experimental
