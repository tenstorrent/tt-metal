// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/offline_kernel_compile.hpp>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/program.hpp>

#include "impl/buffers/circular_buffer.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/kernel_compile_utils.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/genfiles.hpp"
#include "jit_build/jit_build_options.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

#include <hostdevcommon/kernel_structs.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <variant>

namespace tt::tt_metal::experimental {

namespace {

using CBCompileConfig = OfflineKernelCompileParams::CBCompileConfig;

void validate_kernel_config_defines(const std::map<std::string, std::string>& defines) {
    for (const auto& [key, value] : defines) {
        if (value.find('\0') != std::string::npos) {
            throw std::invalid_argument("Define value for key '" + key + "' contains null character");
        }
    }
}

void validate_output_dir(const std::filesystem::path& output_dir) {
    if (output_dir.empty()) {
        throw std::invalid_argument(
            "OfflineKernelCompileParams::output_dir must be non-empty; an empty path would emit "
            "artifacts relative to the process current working directory.");
    }
}

void validate_cb_compile_configs(const std::vector<CBCompileConfig>& cb_compile_configs) {
    std::unordered_set<uint8_t> seen_cb_indices;
    for (const auto& config : cb_compile_configs) {
        if (config.cb_index >= NUM_CIRCULAR_BUFFERS) {
            throw std::invalid_argument(
                "CB compile config has out-of-range cb_index: " + std::to_string(config.cb_index));
        }
        if (config.data_format == DataFormat::Invalid) {
            throw std::invalid_argument(
                "CB compile config has invalid data_format for cb_index: " + std::to_string(config.cb_index));
        }
        if (!seen_cb_indices.insert(config.cb_index).second) {
            throw std::invalid_argument("CB compile config has duplicate cb_index: " + std::to_string(config.cb_index));
        }
    }
}

std::shared_ptr<Kernel> make_offline_kernel(
    const KernelSource& kernel_src,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig>& config) {
    return std::visit(
        [&](const auto& cfg) -> std::shared_ptr<Kernel> {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                return std::make_shared<DataMovementKernel>(kernel_src, core_range_set, cfg);
            } else {
                return std::make_shared<ComputeKernel>(kernel_src, core_range_set, cfg);
            }
        },
        config);
}

void apply_cb_compile_configs(JitBuildOptions& build_options, const std::vector<CBCompileConfig>& cb_compile_configs) {
    for (const auto& cb_compile_config : cb_compile_configs) {
        build_options.set_cb_data_fmt_and_tile(
            static_cast<CBIndex>(cb_compile_config.cb_index), cb_compile_config.data_format, cb_compile_config.tile);
    }
}

// Generate sources and invoke the JIT build for one (kernel, build_env) pair against a *local*
// BuildEnvManager, mirroring the runtime bodies of DataMovementKernel::generate_binaries and
// ComputeKernel::generate_binaries (which are unusable here because they reach into the
// process-wide BuildEnvManager::get_instance() and MetalContext singletons).
//
// TODO: the kernel-type if/else below does not extend to other Kernel subclasses
// (Ethernet, DRAM, Quasar variants); this dispatch should live on Kernel itself.
void generate_kernel_binaries_offline(
    const std::shared_ptr<Kernel>& kernel,
    BuildEnvManager& build_env_manager,
    const DeviceBuildEnv& device_build_env,
    JitBuildOptions& build_options,
    const Hal& hal) {
    const uint32_t programmable_core_type_idx =
        hal.get_programmable_core_type_index(kernel->get_kernel_programmable_core_type());
    const uint32_t processor_class_idx =
        static_cast<std::underlying_type_t<HalProcessorClassType>>(kernel->get_kernel_processor_class());

    jit_build_genfiles_descriptors(device_build_env.build_env, build_options);

    if (kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE) {
        jit_build_genfiles_triscs_src(device_build_env.build_env, *kernel, kernel->kernel_source());
        const auto build_states =
            build_env_manager.get_kernel_build_states(0, programmable_core_type_idx, processor_class_idx);
        jit_build_subset(build_states, kernel.get());
    } else {
        jit_build_genfiles_kernel_include(device_build_env.build_env, *kernel, kernel->kernel_source());
        const uint32_t riscv_id = kernel->get_kernel_processor_type(0);
        const auto& build_state =
            build_env_manager.get_kernel_build_state(0, programmable_core_type_idx, processor_class_idx, riscv_id);
        jit_build(build_state, kernel.get());
    }
}

// Copy each generated ELF for `kernel` from the local build env's kernel root into
// <output_dir>/<kernel_name>/<compile_hash>/<target_full_path>. Both source and destination
// paths are computed by BuildEnvManager::get_kernel_binary_path so the layout matches what
// the runtime precompiled-loader path searches for.
void copy_generated_elfs_to_output_dir(
    const std::shared_ptr<Kernel>& kernel,
    BuildEnvManager& build_env_manager,
    const DeviceBuildEnv& device_build_env,
    const Hal& hal,
    const std::filesystem::path& output_dir) {
    const std::string source_kernel_root = device_build_env.build_env.get_out_kernel_root_path();
    const uint32_t programmable_core_type_idx =
        hal.get_programmable_core_type_index(kernel->get_kernel_programmable_core_type());
    const uint32_t processor_class_idx =
        static_cast<std::underlying_type_t<HalProcessorClassType>>(kernel->get_kernel_processor_class());

    for (uint8_t binary_index = 0; binary_index < kernel->expected_num_binaries(); ++binary_index) {
        const uint32_t processor_id = kernel->get_kernel_processor_type(binary_index);
        const std::string source_elf_path = build_env_manager.get_kernel_binary_path(
            0,
            programmable_core_type_idx,
            processor_class_idx,
            processor_id,
            source_kernel_root,
            kernel->get_full_kernel_name());
        const std::string dest_elf_path = build_env_manager.get_kernel_binary_path(
            0,
            programmable_core_type_idx,
            processor_class_idx,
            processor_id,
            output_dir.string(),
            kernel->get_full_kernel_name());

        std::filesystem::create_directories(std::filesystem::path(dest_elf_path).parent_path());
        std::filesystem::copy_file(source_elf_path, dest_elf_path, std::filesystem::copy_options::overwrite_existing);
    }
}

}  // namespace

std::vector<OfflineKernelCompileParams::CBCompileConfig> CBCompileConfigsFromProgram(
    const Program& program, KernelHandle kernel) {
    const std::shared_ptr<Kernel> kernel_ptr = program.impl().get_kernel(kernel);

    std::map<uint8_t, CBCompileConfig> compile_config_by_cb_index;
    for (const auto& logical_cr : kernel_ptr->logical_coreranges()) {
        const std::vector<std::shared_ptr<CircularBufferImpl>> cbs_on_core_range =
            program.impl().circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core_range) {
            for (const auto buffer_index : circular_buffer->buffer_indices()) {
                const uint8_t cb_index = static_cast<uint8_t>(buffer_index);
                const CBCompileConfig candidate_config{
                    .cb_index = cb_index,
                    .data_format = circular_buffer->data_format(buffer_index),
                    .tile = circular_buffer->tile(buffer_index),
                };
                compile_config_by_cb_index.insert_or_assign(cb_index, candidate_config);
            }
        }
    }

    std::vector<CBCompileConfig> compile_configs;
    compile_configs.reserve(compile_config_by_cb_index.size());
    for (const auto& [_, compile_config] : compile_config_by_cb_index) {
        compile_configs.push_back(compile_config);
    }
    return compile_configs;
}

void CompileKernelOffline(
    const std::string& file_name,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const OfflineKernelCompileParams& params) {
    std::visit([](const auto& cfg) { validate_kernel_config_defines(cfg.defines); }, config);
    validate_cb_compile_configs(params.cb_compile_configs);
    validate_output_dir(params.output_dir);

    // Mode is presently a single-alternative variant (AllSupportedProducts). std::visit + the
    // static_assert below let future alternatives be added without silently bypassing this body.
    std::visit(
        [&](const auto& mode) {
            using ModeT = std::decay_t<decltype(mode)>;
            static_assert(
                std::is_same_v<ModeT, OfflineKernelCompileParams::AllSupportedProducts>,
                "Unhandled OfflineKernelCompileParams::Mode alternative");

            // Build one Kernel instance and reuse it across every enumerated JitDeviceConfig.
            // The kernel's content (source, compile args, defines) is identical per config; the
            // per-config state (build options, full name, hash) is stored on JitBuildOptions and
            // refreshed on the kernel via set_full_name() each iteration.
            const KernelSource kernel_src(file_name, KernelSource::FILE_PATH);
            const CoreRangeSet placeholder_core_range_set(CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}});
            const std::shared_ptr<Kernel> kernel = make_offline_kernel(kernel_src, placeholder_core_range_set, config);

            llrt::RunTimeOptions rtoptions;
            enumerate_offline_compile_device_configs(rtoptions, [&](const JitDeviceConfig& jit_device_config) {
                BuildEnvManager build_env_manager(*jit_device_config.hal);
                build_env_manager.add_build_env(0, jit_device_config, rtoptions);
                const DeviceBuildEnv& device_build_env = build_env_manager.get_device_build_env(0);

                JitBuildOptions build_options(device_build_env.build_env);
                kernel->set_build_options(build_options);
                apply_cb_compile_configs(build_options, params.cb_compile_configs);

                const std::size_t kernel_hash =
                    detail::KernelCompileHash(kernel, build_options, device_build_env.build_key());
                const std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash) + "/";
                kernel->set_full_name(kernel_path_suffix);
                build_options.set_name(kernel_path_suffix);

                generate_kernel_binaries_offline(
                    kernel, build_env_manager, device_build_env, build_options, *jit_device_config.hal);

                copy_generated_elfs_to_output_dir(
                    kernel, build_env_manager, device_build_env, *jit_device_config.hal, params.output_dir);
            });
        },
        params.mode);
}

}  // namespace tt::tt_metal::experimental
