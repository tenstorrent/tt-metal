// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/persistent_kernel_cache.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/utils.hpp>
#include "impl/kernels/kernel_impl.hpp"

using namespace tt::tt_metal;

TEST_F(DeviceFixture, TensixTestIncompleteKernelBinaryWithPersistentCache) {
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";

    for (IDevice* device : this->devices_) {
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

        const auto& kernels = program.get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name = KernelImpl::from(*kernels.at(kernel_handle)).get_full_kernel_name();

        const std::string successful_marker_path =
            build_state.get_out_path() + full_kernel_name + SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME;

        std::filesystem::remove(successful_marker_path);

        const std::string elf_file_path = build_state.get_target_out_path(full_kernel_name);
        const auto t0 = std::filesystem::last_write_time(elf_file_path);

        detail::ClearKernelCache();

        program = CreateProgram();
        kernel_handle = CreateKernel(program, kernel_file, CoreCoord(0, 0), config);
        detail::CompileProgram(device, program);

        const auto t1 = std::filesystem::last_write_time(elf_file_path);

        detail::DisablePersistentKernelCache();

        EXPECT_TRUE(std::filesystem::exists(successful_marker_path));
        EXPECT_TRUE(t1 > t0);
    }
}

TEST_F(DeviceFixture, TensixTestEquivalentDataMovementKernelsWithDifferentProcessors) {
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";

    for (IDevice* device : this->devices_) {
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

        const auto& kernels = program.get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const std::string full_kernel_name_riscv_0 =
            KernelImpl::from(*kernels.at(kernel_handle_riscv_0)).get_full_kernel_name();
        const std::string full_kernel_name_riscv_1 =
            KernelImpl::from(*kernels.at(kernel_handle_riscv_1)).get_full_kernel_name();

        const std::string elf_file_path_riscv_0 = build_state_riscv_0.get_target_out_path(full_kernel_name_riscv_0);
        const std::string elf_file_path_riscv_1 = build_state_riscv_1.get_target_out_path(full_kernel_name_riscv_1);

        EXPECT_TRUE(std::filesystem::exists(elf_file_path_riscv_0));
        EXPECT_TRUE(std::filesystem::exists(elf_file_path_riscv_1));
    }
}
