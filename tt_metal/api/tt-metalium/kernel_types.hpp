// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/util.hpp>

namespace tt::tt_metal {

using KernelHandle = std::uint32_t;

// Option that controls build optimization level
enum class KernelBuildOptLevel : uint8_t {
    O1,     // Turns on level 1 optimization. Same as O.
    O2,     // Turns on level 2 optimization and also all flags specified by O1.
    O3,     // Turns on level 3 optimization and also all flags specified by O2.
    O0,     // Reduce compilation time and make debugging produce the expected results.
    Os,     // Optimize for size and also O2 optimizations except for those that increase binary size.
    Ofast,  // Turns on level O3 and also non standard optimizations.
    Oz,     // Aggresively optimize for size rather than speed.
};

struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;  // For data transfer kernels: NCRISC & BRISC
    NOC noc = NOC::RISCV_0_default;
    NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
    // Both compile_args and named_compile_args contain compile time arguments
    // The former is accessed by index, the latter by name
    // Can be used in new/existing kernels by explicitly defining them in the config
    // Ex. std::vector<uint32_t> compile_args = {5, 7};
    //     std::unordered_map<std::string, uint32_t> named_compile_args = {{"arg1", 5}, {"arg2", 7}};
    //     CreateKernel(program, "kernel.cpp", core, DataMovementConfig{.compile_args = compile_args,
    //     .named_compile_args = named_compile_args})
    std::unordered_map<std::string, uint32_t> named_compile_args;
    // Set the compiler and linker optimization level
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2;
};

struct ReaderDataMovementConfig : public DataMovementConfig {
    ReaderDataMovementConfig(
        std::vector<uint32_t> compile_args = {},
        std::map<std::string, std::string> defines = {},
        std::unordered_map<std::string, uint32_t> named_compile_args = {},
        KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2);
};

struct WriterDataMovementConfig : public DataMovementConfig {
    WriterDataMovementConfig(
        std::vector<uint32_t> compile_args = {},
        std::map<std::string, std::string> defines = {},
        std::unordered_map<std::string, uint32_t> named_compile_args = {},
        KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2);
};

struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
    // Both compile_args and named_compile_args contain compile time arguments
    // The former is accessed by index, the latter by name
    // Can be used in new/existing kernels by explicitly defining them in the config
    // Ex. std::vector<uint32_t> compile_args = {5, 7};
    //     std::unordered_map<std::string, uint32_t> named_compile_args = {{"arg1", 5}, {"arg2", 7}};
    //     CreateKernel(program, "kernel.cpp", core, ComputeConfig{.compile_args = compile_args, .named_compile_args =
    //     named_compile_args})
    std::unordered_map<std::string, uint32_t> named_compile_args;
    // Set the compiler and linker optimization level
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O3;
};

struct EthernetConfig {
    Eth eth_mode = Eth::SENDER;
    NOC noc = NOC::NOC_0;
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
    // Both compile_args and named_compile_args contain compile time arguments
    // The former is accessed by index, the latter by name
    // Can be used in new/existing kernels by explicitly defining them in the config
    // Ex. std::vector<uint32_t> compile_args = {5, 7};
    //     std::unordered_map<std::string, uint32_t> named_compile_args = {{"arg1", 5}, {"arg2", 7}};
    //     CreateKernel(program, "kernel.cpp", core, EthernetConfig{.compile_args = compile_args, .named_compile_args =
    //     named_compile_args})
    std::unordered_map<std::string, uint32_t> named_compile_args;
    // Set the compiler and linker optimization level
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::Os;
};

}  // namespace tt::tt_metal
