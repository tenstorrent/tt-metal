// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <kernel_types.hpp>
#include <stdint.h>
#include "impl/context/metal_context.hpp"
#include <utility>

#include "util.hpp"

namespace tt::tt_metal {

ReaderDataMovementConfig::ReaderDataMovementConfig(
    std::vector<uint32_t> compile_args, std::map<std::string, std::string> defines, KernelBuildOptLevel opt_level) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = detail::GetPreferredNOCForDRAMRead(tt::tt_metal::MetalContext::instance().get_cluster().arch()),
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines),
        .opt_level = opt_level} {}

WriterDataMovementConfig::WriterDataMovementConfig(
    std::vector<uint32_t> compile_args, std::map<std::string, std::string> defines, KernelBuildOptLevel opt_level) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = detail::GetPreferredNOCForDRAMWrite(tt::tt_metal::MetalContext::instance().get_cluster().arch()),
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines),
        .opt_level = opt_level} {}

}  // namespace tt::tt_metal
