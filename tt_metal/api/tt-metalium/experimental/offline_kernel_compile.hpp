// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <cstddef>
#include <vector>

#include <tt-metalium/data_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal {
class Program;
}

namespace tt::tt_metal::experimental {

/**
 * Parameters for offline compilation of a single kernel.
 *
 * The caller provides target-selection mode, output path, and per-CB compile
 * configuration required to reproduce runtime compile-hash inputs.
 *
 * @note Experimental. Will be superseded by the Metal 2.0 offline compilation
 *       API (work in progress).
 */
struct OfflineKernelCompileParams {
    /**
     * Compile-time configuration for one circular buffer index.
     *
     * `cb_index` selects the CB slot, `data_format` sets the CB data format,
     * and `tile` optionally overrides tile shape metadata.
     */
    struct CBCompileConfig {
        uint8_t cb_index = 0;
        DataFormat data_format = DataFormat::Invalid;
        std::optional<Tile> tile = std::nullopt;
    };

    /// Compile for every product/device configuration supported by this API.
    struct AllSupportedProducts {};

    /// Offline compile target-selection mode.
    using Mode = std::variant<AllSupportedProducts>;

    /// Target-selection mode. Default: all supported products.
    Mode mode = AllSupportedProducts{};
    /// Root directory for emitted offline kernel artifacts. Must be non-empty;
    /// otherwise CompileKernelOffline throws std::invalid_argument.
    std::filesystem::path output_dir;
    /// Per-CB compile configuration used to populate compile metadata.
    std::vector<CBCompileConfig> cb_compile_configs;
};

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
 *
 * @note Experimental. Will be superseded by the Metal 2.0 precompiled-kernel
 *       loading API (work in progress).
 */
KernelHandle CreateKernelFromPrecompiled(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const PrecompiledKernelConfig& precompiled_config);

/**
 * Derive per-CB compile configs for one kernel from an existing Program.
 *
 * Walks the kernel's logical core ranges, collects intersecting circular
 * buffers, and returns one normalized entry per CB index used by the kernel.
 *
 * @param program Program containing the kernel and circular buffers.
 * @param kernel  Kernel handle in `program` to derive CB compile configs for.
 * @return Normalized per-CB compile configuration vector.
 *
 * @note Experimental. Will be superseded by the Metal 2.0 offline compilation
 *       API (work in progress).
 */
std::vector<OfflineKernelCompileParams::CBCompileConfig> CBCompileConfigsFromProgram(
    const Program& program, KernelHandle kernel);

/**
 * Compile one kernel offline and emit runtime-compatible artifacts.
 *
 * This API validates compile parameters and writes binaries under
 * `params.output_dir` for the selected compile mode. `params.output_dir` must be non-empty.
 *
 * @param file_name Kernel source file path.
 * @param config    Kernel config: DataMovementConfig or ComputeConfig.
 * @param params    Offline compilation request and CB compile configs.
 *
 * @note Experimental. Will be superseded by the Metal 2.0 offline compilation
 *       API (work in progress).
 */
void CompileKernelOffline(
    const std::string& file_name,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const OfflineKernelCompileParams& params);

}  // namespace tt::tt_metal::experimental
