// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <variant>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

/**
 * The APIs in this file are for initial support of Quasar, our next-generation architecture.
 * Quasar has significant architectural differences from Wormhole and Blackhole. Some key differences are:
 * - There are 8 data movement cores per cluster
 * - There are 4 compute cores per cluster with each compute core having 4 TRISC processors
 * - Each cluster contains 4 MB of L1 SRAM which is shared by all the cores in the cluster
 * - Users will target clusters instead of the cores within the clusters; our internal implementation will choose which
 *   cores on each cluster are used
 * These APIs are very experimental and will evolve accordingly over time.
 */

namespace tt::tt_metal {
class Program;

namespace experimental::quasar {
static constexpr uint32_t QUASAR_NUM_DM_CORES_PER_CLUSTER = 8;
static constexpr uint32_t QUASAR_NUM_COMPUTE_CORES_PER_CLUSTER = 4;

struct QuasarDataMovementConfig {
    // Number of data movement cores per cluster to use
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

struct QuasarComputeConfig {
    // Number of compute cores per cluster to use
    uint32_t num_processors_per_cluster = QUASAR_NUM_COMPUTE_CORES_PER_CLUSTER;

    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;

    std::vector<uint32_t> compile_args;

    std::map<std::string, std::string> defines;

    // Both compile_args and named_compile_args contain compile time arguments
    // The former is accessed by index, the latter by name
    // Can be used in new/existing kernels by explicitly defining them in the config
    // Ex. std::vector<uint32_t> compile_args = {5, 7};
    //     std::unordered_map<std::string, uint32_t> named_compile_args = {{"arg1", 5}, {"arg2", 7}};
    //     CreateKernel(program, "kernel.cpp", core, QuasarComputeConfig{.compile_args = compile_args,
    //     .named_compile_args = named_compile_args})
    std::unordered_map<std::string, uint32_t> named_compile_args;
    // Set the compiler and linker optimization level
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O3;
};

/**
 * @brief Create a data movement kernel and add it to the program.
 *
 * @param program The program to which this kernel will be added to
 * @param file_name Path to kernel source file
 * @param core_spec Either a single logical cluster, a range of logical clusters or a set of logical cluster ranges that
 * indicate which clusters the kernel is placed on
 * @param config Config for data movement kernel
 * @return Kernel ID
 */
KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const QuasarDataMovementConfig& config);

/**
 * @brief Create a compute kernel and add it to the program.
 *
 * @param program The program to which this kernel will be added to
 * @param file_name Path to kernel source file
 * @param core_spec Either a single logical cluster, a range of logical clusters or a set of logical cluster ranges that
 * indicate which clusters the kernel is placed on
 * @param config Config for compute kernel
 * @return Kernel ID
 */
KernelHandle CreateKernel(
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const QuasarComputeConfig& config);
}  // namespace experimental::quasar
}  // namespace tt::tt_metal
