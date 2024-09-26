// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/base_types.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <map>
#include <vector>
#include <string>

namespace tt::tt_metal {

using KernelHandle = std::uint16_t;

struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;  // For data transfer kernels: NCRISC & BRISC
    NOC noc = NOC::RISCV_0_default;
    NOC_MODE noc_mode = NOC_MODE::DEDICATED_NOC_PER_DM;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
};

struct ReaderDataMovementConfig : public DataMovementConfig {
    ReaderDataMovementConfig(std::vector<uint32_t> compile_args = {}, std::map<std::string, std::string> defines = {}) :
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = detail::GetPreferredNOCForDRAMRead(tt::Cluster::instance().arch()),
            .noc_mode = NOC_MODE::DEDICATED_NOC_PER_DM,
            .compile_args = compile_args,
            .defines = defines} {}
};

struct WriterDataMovementConfig : public DataMovementConfig {
    WriterDataMovementConfig(std::vector<uint32_t> compile_args = {}, std::map<std::string, std::string> defines = {}) :
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = detail::GetPreferredNOCForDRAMWrite(tt::Cluster::instance().arch()),
            .noc_mode = NOC_MODE::DEDICATED_NOC_PER_DM,
            .compile_args = compile_args,
            .defines = defines} {}
};

struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
};

struct EthernetConfig {
    Eth eth_mode = Eth::SENDER;
    NOC noc = NOC::NOC_0;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;

};

} // namespace tt::tt_metal
