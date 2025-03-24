// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <unordered_set>

#include "core_coord.hpp"
#include "detail/kernel_cache.hpp"
#include "device.hpp"
#include "device_fixture.hpp"
#include "hal.hpp"
#include "host_api.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "kernel.hpp"
#include "kernel_types.hpp"
#include "persistent_kernel_cache.hpp"
#include "tt_metal.hpp"

using namespace tt::tt_metal;

TEST_F(DeviceFixture, TensixTestIncompleteKernelBinaryWithPersistentCache) {
    const string kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";

    for (IDevice* device : this->devices_) {
        detail::ClearKernelCache();
        detail::EnablePersistentKernelCache();

        Program program;
        KernelHandle kernel_handle;
        const DataMovementConfig config = {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};

        program = CreateProgram();
        kernel_handle = CreateKernel(program, kernel_file, CoreCoord(0, 0), config);
        detail::CompileProgram(device, program);

        const uint32_t tensix_core_type = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const uint32_t dm_class_idx = magic_enum::enum_integer(HalProcessorClassType::DM);
        const int riscv_id = static_cast<std::underlying_type<DataMovementProcessor>::type>(config.processor);
        const JitBuildState& build_state = BuildEnvManager::get_instance().get_kernel_build_state(
            device->build_id(), tensix_core_type, dm_class_idx, riscv_id);

        const auto& kernels = program.get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::TENSIX));
        const string full_kernel_name = kernels.at(kernel_handle)->get_full_kernel_name();

        const string successful_marker_path =
            build_state.get_out_path() + full_kernel_name + SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME;

        tt::log_info("Deleting .SUCCESS file");
        std::filesystem::remove(successful_marker_path);

        const string elf_file_path = build_state.get_target_out_path(full_kernel_name);
        const auto t0 = std::filesystem::last_write_time(elf_file_path);

        detail::ClearKernelCache();

        program = CreateProgram();
        kernel_handle = CreateKernel(program, kernel_file, CoreCoord(0, 0), config);
        detail::CompileProgram(device, program);

        detail::DisablePersistentKernelCache();

        const auto t1 = std::filesystem::last_write_time(elf_file_path);

        EXPECT_TRUE(std::filesystem::exists(successful_marker_path));
        EXPECT_TRUE(t1 > t0);
    }
}
