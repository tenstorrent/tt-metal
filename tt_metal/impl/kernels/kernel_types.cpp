// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <kernel_types.hpp>
#include <stdint.h>
#include "base_types.hpp"
#include "circular_buffer_constants.h"
#include "impl/context/metal_context.hpp"
#include <utility>


namespace tt::tt_metal {

ReaderDataMovementConfig::ReaderDataMovementConfig(
    std::vector<uint32_t> compile_args,
    std::map<std::string, std::string> defines,
    std::unordered_map<std::string, uint32_t> named_compile_args,
    KernelBuildOptLevel opt_level) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = detail::preferred_noc_for_dram_read(tt::tt_metal::MetalContext::instance().get_cluster().arch()),
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines),
        .named_compile_args = std::move(named_compile_args),
        .opt_level = opt_level} {}

WriterDataMovementConfig::WriterDataMovementConfig(
    std::vector<uint32_t> compile_args,
    std::map<std::string, std::string> defines,
    std::unordered_map<std::string, uint32_t> named_compile_args,
    KernelBuildOptLevel opt_level) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = detail::preferred_noc_for_dram_write(tt::tt_metal::MetalContext::instance().get_cluster().arch()),
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines),
        .named_compile_args = std::move(named_compile_args),
        .opt_level = opt_level} {}

std::vector<UnpackToDestMode> ComputeConfig::default_unpack_to_dest_modes() {
    static std::vector<UnpackToDestMode> default_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    // Explicitly a copy
    return std::vector<UnpackToDestMode>(default_modes);
}

}  // namespace tt::tt_metal
