// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <cstdint>
#include <map>
#include <string_view>
#include <unordered_map>
#include <string>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {

// 341 = (4096/(3 * sizeof(uint32_t)), where
// - 4096 - packet size in dispatch
// - 3 - number of kernels per tensix
constexpr uint32_t max_runtime_args = 341;

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

// These are only used in op_profiler, are unstable and have not been designed for general use.
namespace detail {

struct KernelBinaryMeta {
    // This maps to Kernel::get_kernel_processor_type
    using ProcessorType = uint32_t;
    ProcessorType processor_type;
    std::size_t packed_size;
};

struct KernelMeta {
    // Kernel identifiers:
    // Owner for name and source is the Program class,
    // (direct owner is the Kernel class internally)
    std::string_view name, source;

    // Core identifiers:
    HalProcessorClassType processor_class;
    HalProgrammableCoreType programmable_core_type;

    // Core configuration:
    // This optinonal only contains a MathFidelity if the kernel is a compute kernel
    std::optional<MathFidelity> math_fidelity;

    // Binary metadata:
    std::vector<KernelBinaryMeta> binary_meta;
};

}  // namespace detail

}  // namespace tt::tt_metal
