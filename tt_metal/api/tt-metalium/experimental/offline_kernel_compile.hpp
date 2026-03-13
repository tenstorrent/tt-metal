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

struct PrecompiledKernelConfig {
    enum class BinaryNotFoundPolicy {
        JitCompile,
        Error,
    };

    std::string precompiled_dir;
    BinaryNotFoundPolicy fallback_policy = BinaryNotFoundPolicy::Error;
};

class PrecompiledKernelNotFoundError : public std::runtime_error {
public:
    PrecompiledKernelNotFoundError(
        std::string kernel_name,
        size_t compile_hash,
        std::string precompiled_dir,
        PrecompiledKernelConfig::BinaryNotFoundPolicy fallback_policy) :
        std::runtime_error("Precompiled kernel binaries not found."),
        kernel_name_(std::move(kernel_name)),
        compile_hash_(compile_hash),
        precompiled_dir_(std::move(precompiled_dir)),
        fallback_policy_(fallback_policy) {}

    const std::string& kernel_name() const noexcept { return kernel_name_; }
    size_t compile_hash() const noexcept { return compile_hash_; }
    const std::string& precompiled_dir() const noexcept { return precompiled_dir_; }
    PrecompiledKernelConfig::BinaryNotFoundPolicy fallback_policy() const noexcept { return fallback_policy_; }

private:
    std::string kernel_name_;
    size_t compile_hash_;
    std::string precompiled_dir_;
    PrecompiledKernelConfig::BinaryNotFoundPolicy fallback_policy_;
};

KernelHandle CreateKernelFromPrecompiled(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const PrecompiledKernelConfig& precompiled_config);

}  // namespace tt::tt_metal::experimental
