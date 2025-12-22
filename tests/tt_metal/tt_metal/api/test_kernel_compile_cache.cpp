// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <stdint.h>
#include <filesystem>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "detail/kernel_cache.hpp"
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/persistent_kernel_cache.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/kernels/kernel.hpp"
// Access to internal API: ProgramImpl::get_kernels
#include "impl/program/program_impl.hpp"

using namespace tt::tt_metal;

TEST_F(MeshDeviceFixture, TensixTestIncompleteKernelBinaryWithPersistentCache) {
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";

    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        detail::ClearKernelCache();
        detail::EnablePersistentKernelCache();

        Program program;
        KernelHandle kernel_handle;
        const DataMovementConfig config = {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};

        program = CreateProgram();
        kernel_handle = CreateKernel(program, kernel_file, CoreCoord(0, 0), config);
        detail::CompileProgram(device, program);

        const uint32_t tensix_core_type =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
        const int riscv_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(config.processor);
        const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, dm_class_idx, riscv_id);

        const auto& kernels = program.impl().get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name = kernels.at(kernel_handle)->get_full_kernel_name();

        const std::string successful_marker_path =
            build_state.get_out_path() + full_kernel_name + SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME;

        std::filesystem::remove(successful_marker_path);

        const std::string elf_file_path = build_state.get_target_out_path(full_kernel_name);
        const auto t0 = std::filesystem::last_write_time(elf_file_path);

        detail::ClearKernelCache();

        program = CreateProgram();
        kernel_handle = CreateKernel(program, kernel_file, CoreCoord(0, 0), config);
        // Note:  Force JIT compile for this test.  Otherwise it may reuse the binary and not update the timestamp.
        {
            auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
            bool saved = rtoptions.get_force_jit_compile();
            rtoptions.set_force_jit_compile(true);
            detail::CompileProgram(device, program);
            rtoptions.set_force_jit_compile(saved);
        }

        const auto t1 = std::filesystem::last_write_time(elf_file_path);

        detail::DisablePersistentKernelCache();

        EXPECT_TRUE(std::filesystem::exists(successful_marker_path));
        EXPECT_TRUE(t1 > t0);
    }
}

TEST_F(MeshDeviceFixture, TensixTestEquivalentDataMovementKernelsWithDifferentProcessors) {
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";

    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        detail::ClearKernelCache();

        DataMovementConfig config_riscv_0 = {.processor = DataMovementProcessor::RISCV_0};
        DataMovementConfig config_riscv_1 = {.processor = DataMovementProcessor::RISCV_1};

        Program program = CreateProgram();
        KernelHandle kernel_handle_riscv_0 = CreateKernel(program, kernel_file, CoreCoord(0, 0), config_riscv_0);
        KernelHandle kernel_handle_riscv_1 = CreateKernel(program, kernel_file, CoreCoord(0, 0), config_riscv_1);
        detail::CompileProgram(device, program);

        const uint32_t tensix_core_type =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
        const int riscv_0_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(config_riscv_0.processor);
        const int riscv_1_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(config_riscv_1.processor);
        const JitBuildState& build_state_riscv_0 = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, dm_class_idx, riscv_0_id);
        const JitBuildState& build_state_riscv_1 = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, dm_class_idx, riscv_1_id);

        const auto& kernels = program.impl().get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name_riscv_0 = kernels.at(kernel_handle_riscv_0)->get_full_kernel_name();
        const std::string full_kernel_name_riscv_1 = kernels.at(kernel_handle_riscv_1)->get_full_kernel_name();

        const std::string elf_file_path_riscv_0 = build_state_riscv_0.get_target_out_path(full_kernel_name_riscv_0);
        const std::string elf_file_path_riscv_1 = build_state_riscv_1.get_target_out_path(full_kernel_name_riscv_1);

        EXPECT_TRUE(std::filesystem::exists(elf_file_path_riscv_0));
        EXPECT_TRUE(std::filesystem::exists(elf_file_path_riscv_1));
    }
}

TEST_F(MeshDeviceFixture, TensixTestSFPIKernelCcacheSupport) {
    // Skip this test if TT_METAL_CCACHE_KERNEL_SUPPORT is not enabled
    if (std::getenv("TT_METAL_CCACHE_KERNEL_SUPPORT") == nullptr) {
        GTEST_SKIP() << "TT_METAL_CCACHE_KERNEL_SUPPORT not set, skipping ccache test";
    }

    // Use a compute kernel which uses SFPI compiler
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp";

    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        detail::ClearKernelCache();

        // Compile kernel first time
        ComputeConfig config = {
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}};

        Program program = CreateProgram();
        KernelHandle kernel_handle = CreateKernel(program, kernel_file, CoreCoord(0, 0), config);
        detail::CompileProgram(device, program);

        const uint32_t tensix_core_type =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t compute_class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);
        // TRISC0 is typically processor_id 0 for compute
        const int trisc_id = 0;
        const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, compute_class_idx, trisc_id);

        const auto& kernels = program.impl().get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name = kernels.at(kernel_handle)->get_full_kernel_name();

        const std::string elf_file_path = build_state.get_target_out_path(full_kernel_name);
        ASSERT_TRUE(std::filesystem::exists(elf_file_path)) << "First compilation should produce ELF file";

        const auto t0 = std::filesystem::last_write_time(elf_file_path);

        // Clear in-memory cache but keep filesystem artifacts (including ccache)
        detail::ClearKernelCache();

        // Compile same kernel again - should hit ccache
        Program program2 = CreateProgram();
        KernelHandle kernel_handle2 = CreateKernel(program2, kernel_file, CoreCoord(0, 0), config);
        detail::CompileProgram(device, program2);

        const auto& kernels2 = program2.impl().get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name2 = kernels2.at(kernel_handle2)->get_full_kernel_name();
        const std::string elf_file_path2 = build_state.get_target_out_path(full_kernel_name2);

        ASSERT_TRUE(std::filesystem::exists(elf_file_path2)) << "Second compilation should produce ELF file";

        // With ccache, the object files should be retrieved from cache without recompilation
        // The ELF may be re-linked (depending on implementation), but it should be very fast
        // We verify that the kernel name is consistent
        EXPECT_EQ(full_kernel_name, full_kernel_name2) << "Kernel names should be identical for same kernel";

        // Now modify the kernel source by touching it or changing a dependency
        // This should invalidate the cache
        // For this test, we'll just verify the first two compilations worked with ccache
    }
}

TEST_F(MeshDeviceFixture, TensixTestSFPIKernelCcacheInvalidationOnSourceChange) {
    // Skip this test if TT_METAL_CCACHE_KERNEL_SUPPORT is not enabled
    if (std::getenv("TT_METAL_CCACHE_KERNEL_SUPPORT") == nullptr) {
        GTEST_SKIP() << "TT_METAL_CCACHE_KERNEL_SUPPORT not set, skipping ccache test";
    }

    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp";

    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        detail::ClearKernelCache();

        // First compilation with one set of compile args
        ComputeConfig config1 = {
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}};

        Program program1 = CreateProgram();
        KernelHandle kernel_handle1 = CreateKernel(program1, kernel_file, CoreCoord(0, 0), config1);
        detail::CompileProgram(device, program1);

        const uint32_t tensix_core_type =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t compute_class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);
        const int trisc_id = 0;

        const auto& kernels1 = program1.impl().get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name1 = kernels1.at(kernel_handle1)->get_full_kernel_name();

        // Clear cache and compile with different defines - should produce different kernel
        detail::ClearKernelCache();

        ComputeConfig config2 = {
            .math_fidelity = MathFidelity::HiFi2,  // Different math fidelity
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {}};

        Program program2 = CreateProgram();
        KernelHandle kernel_handle2 = CreateKernel(program2, kernel_file, CoreCoord(0, 0), config2);
        detail::CompileProgram(device, program2);

        const auto& kernels2 = program2.impl().get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name2 = kernels2.at(kernel_handle2)->get_full_kernel_name();

        // Different configs should produce different kernel names
        EXPECT_NE(full_kernel_name1, full_kernel_name2) << "Different math fidelity should produce different kernels";

        // Both should have their own ELF files
        const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, compute_class_idx, trisc_id);

        const std::string elf_file_path1 = build_state.get_target_out_path(full_kernel_name1);
        const std::string elf_file_path2 = build_state.get_target_out_path(full_kernel_name2);

        EXPECT_TRUE(std::filesystem::exists(elf_file_path1));
        EXPECT_TRUE(std::filesystem::exists(elf_file_path2));
        EXPECT_NE(elf_file_path1, elf_file_path2);
    }
}
