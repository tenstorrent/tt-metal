// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/base_types.hpp"
#include <map>
#include <vector>
#include <string>

namespace tt::tt_metal {

using KernelHandle = std::uint16_t;

enum class DataMovementProcessor {
    RISCV_0 = 0,  // BRISC
    RISCV_1 = 1,  // NCRISC
};

enum NOC : uint8_t {
    RISCV_0_default = 0,
    RISCV_1_default = 1,
    NOC_0 = 0,
    NOC_1 = 1,
};

enum Eth : uint8_t {
    SENDER = 0,
    RECEIVER = 1,
};

struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;  // For data transfer kernels: NCRISC & BRISC
    NOC noc = NOC::RISCV_0_default;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
};

struct EthernetConfig {
    Eth eth_mode = Eth::SENDER;
    NOC noc = NOC::NOC_0; // TODO: is this needed?
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
};

struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
};


} // namespace tt::tt_metal
