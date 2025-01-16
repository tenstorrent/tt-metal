// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <kernel_types.hpp>
#include <tt_cluster.hpp>

#include <utility>

namespace tt::tt_metal {

ReaderDataMovementConfig::ReaderDataMovementConfig(
    std::vector<uint32_t> compile_args, std::map<std::string, std::string> defines) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch()),
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines)} {}

WriterDataMovementConfig::WriterDataMovementConfig(
    std::vector<uint32_t> compile_args, std::map<std::string, std::string> defines) :
    DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch()),
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
        .compile_args = std::move(compile_args),
        .defines = std::move(defines)} {}

}  // namespace tt::tt_metal
