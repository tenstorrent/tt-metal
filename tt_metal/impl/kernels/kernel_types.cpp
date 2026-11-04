// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <kernel_types.hpp>
#include <cstdint>
#include <utility>

namespace tt::tt_metal {

ReaderDataMovementConfig::ReaderDataMovementConfig(
    std::vector<uint32_t> compile_args,
    std::map<std::string, std::string> defines,
    std::unordered_map<std::string, uint32_t> named_compile_args,
    KernelBuildOptLevel opt_level,
    std::vector<std::filesystem::path> compiler_include_paths) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        // Metal 1.0 only supports WH and BH, for which the preferred DRAM-read NOC is NOC_0.
        // Hardcoded to avoid a dependency on MetalContext; this API is slated for deprecation.
        .noc = NOC::NOC_0,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines),
        .named_compile_args = std::move(named_compile_args),
        .opt_level = opt_level,
        .compiler_include_paths = std::move(compiler_include_paths)} {}

WriterDataMovementConfig::WriterDataMovementConfig(
    std::vector<uint32_t> compile_args,
    std::map<std::string, std::string> defines,
    std::unordered_map<std::string, uint32_t> named_compile_args,
    KernelBuildOptLevel opt_level,
    std::vector<std::filesystem::path> compiler_include_paths) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        // Metal 1.0 only supports WH and BH, for which the preferred DRAM-write NOC is NOC_1.
        // Hardcoded to avoid a dependency on MetalContext; this API is slated for deprecation.
        .noc = NOC::NOC_1,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines),
        .named_compile_args = std::move(named_compile_args),
        .opt_level = opt_level,
        .compiler_include_paths = std::move(compiler_include_paths)} {}

}  // namespace tt::tt_metal
