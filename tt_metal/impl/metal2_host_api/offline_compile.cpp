// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/offline_compile.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>                              // detail::CompileProgram
#include <tt-metalium/experimental/metal2_host_api/program.hpp>  // MakeProgramFromSpec

#include "impl/context/metal_context.hpp"
#include "impl/jit_server/jit_compile_rpc_client.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build_env_manager.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_metal::experimental {

namespace {

// Iterate every kernel in `program_impl`, in programmable-core-type order.
//
// NOTE: get_kernels(idx) returns the per-core-type map; num_kernels() is a total count and is
// NOT an index space, so it must not be used to drive this loop. The number of programmable
// core types comes from the HAL (matching how ProgramImpl sizes its kernels_ vector).
template <typename Fn>
void for_each_kernel(detail::ProgramImpl& program_impl, const Fn& fn) {
    const uint32_t core_type_count = MetalContext::instance().hal().get_programmable_core_type_count();
    for (uint32_t core_type_idx = 0; core_type_idx < core_type_count; ++core_type_idx) {
        for (auto& [kernel_handle, kernel] : program_impl.get_kernels(core_type_idx)) {
            fn(kernel);
        }
    }
}

// Copy every generated ELF for `kernel` from the device's JIT kernel root into `output_dir`.
// Both source and destination paths are computed by BuildEnvManager::get_kernel_binary_path so the
// emitted layout matches exactly what the runtime precompiled-loader path searches for.
void copy_kernel_elfs_to_output_dir(
    const std::shared_ptr<Kernel>& kernel,
    BuildEnvManager& build_env_manager,
    ChipId build_id,
    const Hal& hal,
    const std::string& source_kernel_root,
    const std::string& output_dir) {
    const uint32_t programmable_core_type_idx =
        hal.get_programmable_core_type_index(kernel->get_kernel_programmable_core_type());
    const uint32_t processor_class_idx =
        static_cast<std::underlying_type_t<HalProcessorClassType>>(kernel->get_kernel_processor_class());

    for (uint8_t binary_index = 0; binary_index < kernel->expected_num_binaries(); ++binary_index) {
        const uint32_t processor_id = kernel->get_kernel_processor_type(binary_index);
        const std::string source_elf_path = build_env_manager.get_kernel_binary_path(
            build_id,
            programmable_core_type_idx,
            processor_class_idx,
            processor_id,
            source_kernel_root,
            kernel->get_full_kernel_name());
        const std::string dest_elf_path = build_env_manager.get_kernel_binary_path(
            build_id,
            programmable_core_type_idx,
            processor_class_idx,
            processor_id,
            output_dir,
            kernel->get_full_kernel_name());

        std::filesystem::create_directories(std::filesystem::path(dest_elf_path).parent_path());
        std::filesystem::copy_file(source_elf_path, dest_elf_path, std::filesystem::copy_options::overwrite_existing);
    }
}

}  // namespace

void CompileProgramSpecOffline(
    const distributed::MeshDevice& mesh_device, const ProgramSpec& spec, const std::string& output_dir) {
    TT_FATAL(
        !output_dir.empty(),
        "CompileProgramSpecOffline: output_dir must be non-empty; an empty string would emit artifacts "
        "relative to the process current working directory.");
    TT_FATAL(
        !jit_server::JitCompileRpcClient::enabled(),
        "CompileProgramSpecOffline is not supported when the remote JIT compile server is enabled "
        "(TT_METAL_JIT_SERVER_ENABLE). The remote compile path does not emit local ELFs for AOT load.");

    const std::vector<IDevice*> devices = mesh_device.get_devices();
    TT_FATAL(!devices.empty(), "CompileProgramSpecOffline: mesh_device has no devices.");
    if (devices.size() > 1) {
        log_warning(
            tt::LogBuildKernels,
            "CompileProgramSpecOffline: mesh_device has {} devices; only the first device's build env is used "
            "for offline compilation.",
            devices.size());
    }
    // CompileProgram needs a mutable IDevice*, but we only hold a const MeshDevice&; get_devices()
    // is const-qualified yet returns non-const IDevice*, so it is how we obtain a device to compile
    // on. Compiling on devices[0] aligns with the runtime JIT path, which compiles each program
    // once against MeshDevice::build_id() (== reference_device() == devices[0]), not per chip.
    IDevice* device = devices[0];

    // Compile via the real on-device CompileProgram path: this writes ELFs to the JIT cache and
    // sets each kernel's full_name to "<name>/<hash>/" (the suffix the loader recomputes at
    // dispatch time).
    Program program = MakeProgramFromSpec(mesh_device, spec);
    detail::CompileProgram(device, program);

    auto& build_env_manager = BuildEnvManager::get_instance();
    const DeviceBuildEnv& device_build_env = build_env_manager.get_device_build_env(device->build_id());
    const std::string source_kernel_root = device_build_env.build_env.get_out_kernel_root_path();
    const Hal& hal = MetalContext::instance().hal();

    for_each_kernel(program.impl(), [&](const std::shared_ptr<Kernel>& kernel) {
        copy_kernel_elfs_to_output_dir(
            kernel, build_env_manager, device->build_id(), hal, source_kernel_root, output_dir);
    });
}

void SetProgramPrecompiledConfig(Program& program, const PrecompiledKernelConfig& config) {
    TT_FATAL(
        !jit_server::JitCompileRpcClient::enabled(),
        "SetProgramPrecompiledConfig is not supported when the remote JIT compile server is enabled "
        "(TT_METAL_JIT_SERVER_ENABLE). The remote compile path does not honor precompiled config.");
    TT_FATAL(
        !program.impl().is_compiled(),
        "SetProgramPrecompiledConfig must be called before the program is compiled. ProgramImpl::compile "
        "short-circuits once a build key has been compiled, so a late call would silently be a no-op.");

    for_each_kernel(
        program.impl(), [&](const std::shared_ptr<Kernel>& kernel) { kernel->set_precompiled_config(config); });
}

}  // namespace tt::tt_metal::experimental
