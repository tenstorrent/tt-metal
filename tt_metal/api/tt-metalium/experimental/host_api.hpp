// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <variant>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace tt::tt_metal {
class Program;

namespace experimental {
static constexpr uint32_t QUASAR_NUM_DM_CORES_PER_CLUSTER = 8;

struct QuasarDataMovementConfig {
    uint32_t num_processors_per_cluster = QUASAR_NUM_DM_CORES_PER_CLUSTER;

    std::vector<uint32_t> compile_args;

    std::map<std::string, std::string> defines;

    // Both compile_args and named_compile_args contain compile time arguments
    // The former is accessed by index, the latter by name
    // Can be used in new/existing kernels by explicitly defining them in the config
    // Ex. std::vector<uint32_t> compile_args = {5, 7};
    //     std::unordered_map<std::string, uint32_t> named_compile_args = {{"arg1", 5}, {"arg2", 7}};
    //     CreateKernel(program, "kernel.cpp", core, QuasarDataMovementConfig{.compile_args = compile_args,
    //     .named_compile_args = named_compile_args})
    std::unordered_map<std::string, uint32_t> named_compile_args;
    // Set the compiler and linker optimization level
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2;
};

KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const QuasarDataMovementConfig& config);
}
}  // namespace tt::tt_metal
