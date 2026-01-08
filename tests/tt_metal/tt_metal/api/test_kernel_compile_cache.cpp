// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <cstdint>
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
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/kernels/kernel.hpp"
// Access to internal API: ProgramImpl::get_kernels
#include "impl/program/program_impl.hpp"

using namespace tt::tt_metal;

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
        const int riscv_0_id = static_cast<std::underlying_type_t<DataMovementProcessor>>(config_riscv_0.processor);
        const int riscv_1_id = static_cast<std::underlying_type_t<DataMovementProcessor>>(config_riscv_1.processor);
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
