// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "llrt/core_descriptor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "base_types.hpp"
#include "compile_program_with_kernel_path_env_var_fixture.hpp"
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "impl/kernels/kernel_impl.hpp"
#include "impl/program/program_impl.hpp"
#include <tt-metalium/allocator.hpp>

namespace tt::tt_metal {

using namespace tt;

// Ensures we can successfully create kernels on available compute grid
TEST_F(MeshDispatchFixture, TensixCreateKernelsOnComputeCores) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        auto mesh_device = this->devices_.at(id);
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
        EXPECT_NO_THROW(
            tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}););
    }
}

TEST_F(MeshDispatchFixture, DISABLED_TensixIdleEthCreateKernelsOnDispatchCores) {
    if (this->IsSlowDispatch()) {
        GTEST_SKIP() << "This test is only supported in fast dispatch mode";
    }
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        auto mesh_device = this->devices_.at(id);
        auto device = mesh_device->get_devices()[0];

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        const auto& dispatch_core_config = get_dispatch_core_config();
        CoreType dispatch_core_type = dispatch_core_config.get_core_type();
        std::vector<CoreCoord> dispatch_cores =
            tt::get_logical_dispatch_cores(device->id(), device->num_hw_cqs(), dispatch_core_config);
        std::set<CoreRange> dispatch_core_ranges;
        for (CoreCoord core : dispatch_cores) {
            dispatch_core_ranges.emplace(core);
        }
        CoreRangeSet dispatch_core_range_set(dispatch_core_ranges);
        if (dispatch_core_type == CoreType::WORKER) {
            EXPECT_ANY_THROW(tt_metal::CreateKernel(
                                 program_,
                                 "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                                 CoreRangeSet(dispatch_core_range_set),
                                 DataMovementConfig{
                                     .processor = tt_metal::DataMovementProcessor::RISCV_0,
                                     .noc = tt_metal::NOC::RISCV_0_default}););
        } else if (dispatch_core_type == CoreType::ETH) {
            EXPECT_ANY_THROW(tt_metal::CreateKernel(
                                 program_,
                                 "tests/tt_metal/tt_metal/test_kernels/misc/erisc_print.cpp",
                                 CoreRangeSet(dispatch_core_range_set),
                                 EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0}););
        }
    }
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixKernelUnderMetalRootDir) {
    const std::string& kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    create_kernel(kernel_file);
    detail::CompileProgram(this->device_, this->program_);
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixKernelUnderKernelRootDir) {
    const std::string& orig_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    const std::string& new_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/new_kernel.cpp";
    this->setup_kernel_dir(orig_kernel_file, new_kernel_file);
    this->create_kernel(new_kernel_file);
    detail::CompileProgram(this->device_, this->program_);
    this->cleanup_kernel_dir();
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixKernelUnderMetalRootDirAndKernelRootDir) {
    const std::string& kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    this->setup_kernel_dir(kernel_file, kernel_file);
    this->create_kernel(kernel_file);
    detail::CompileProgram(this->device_, this->program_);
    this->cleanup_kernel_dir();
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixNonExistentKernel) {
    const std::string& kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/non_existent_kernel.cpp";
    EXPECT_THROW(this->create_kernel(kernel_file), std::exception);
}

// Unit test for Kernel::compute_hash() - tests that different unpack_to_dest_mode produces different hashes
TEST_F(CompileProgramWithKernelPathEnvVarFixture, TensixTestDifferentUnpackToDestModeShouldProduceDifferentHashes) {
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp";

    // Create default compute config
    ComputeConfig config_default;

    // Create specially configured unpack_to_dest_mode
    ComputeConfig config_fp32(config_default);
    config_fp32.unpack_to_dest_mode.push_back(UnpackToDestMode::UnpackToDestFp32);  // Change CB 0 to FP32 mode

    // Create programs (needed for kernel creation context)
    auto program_default = CreateProgram();
    auto program_fp32 = CreateProgram();

    // Create kernel handles for both configurations
    auto kernel_handle_default = CreateKernel(program_default, kernel_file, CoreCoord(0, 0), config_default);
    auto kernel_handle_fp32 = CreateKernel(program_fp32, kernel_file, CoreCoord(0, 0), config_fp32);

    // Get the kernels from programs
    auto kernel_default = program_default.impl().get_kernel(kernel_handle_default);
    auto kernel_fp32 = program_fp32.impl().get_kernel(kernel_handle_fp32);

    // Direct hash comparison - this tests the actual Kernel::compute_hash() method
    auto hash_default = kernel_default->compute_hash();
    auto hash_fp32 = kernel_fp32->compute_hash();

    // The hashes should be different across two kernels due to the difference in unpack_to_dest_mode
    EXPECT_NE(hash_default, hash_fp32)
        << "unpack_to_dest_mode is not accounted for in computing ComputeKernel::config_hash()";
}

// Testing experimental CreateKernelFromBinary function.
// All steps required are in this test, even though not all are required for a binary kernel.
// 1. Run a program with the kernels to generate the binaries.
// 2. Call the function ComputeKernelOriginalPathHash to get the hash of the original kernel file.
// 3. Set the binary path prefix for the device and call CreateKernelFromBinary for each kernel.
// This test uses a precompiled binary for the kernel for step 3, instead of copying from the cache.
// But the method is the same.
TEST_F(MeshDispatchFixture, TestCreateKernelFromBinary) {
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/compute/simple_add.cpp";
    std::string binary_kernel_path;
    
    for (const auto& mesh_device : this->devices_) {
        CoreCoord core = {0, 0};
        CoreCoord binary_core = {1, 1};
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord);
        distributed::MeshWorkload workload;
        distributed::MeshWorkload binary_workload;

        // Prepare to run the original kernel.
        Program program;
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);
        auto device = mesh_device->get_devices()[0];

        if (device->arch() == tt::ARCH::WORMHOLE_B0) {
            binary_kernel_path = "tests/tt_metal/tt_metal/api/simple_add_binaries/wormhole/kernels";
        } else if (device->arch() == tt::ARCH::BLACKHOLE) {
            binary_kernel_path = "tests/tt_metal/tt_metal/api/simple_add_binaries/blackhole/kernels";
        } else {
            TT_THROW("arch not supported");
        }

        const uint32_t table_address = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
        uint32_t input_a = 1;
        uint32_t input_b = 2;

        // 1. Run the original kernel to generate the binary.
        auto kernel_handle = CreateKernel(program_, kernel_file, core,  tt_metal::DataMovementConfig{ .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = { }});
        SetRuntimeArgs(program_, kernel_handle, core, {table_address, input_a, input_b});
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        std::vector<uint32_t> result;
        tt_metal::detail::ReadFromDeviceL1(device, core, table_address, sizeof(uint32_t), result);
        EXPECT_EQ(result[0], input_a + input_b);

        // Now prepare to run the binary kernel.
        Program binary_program;
        binary_workload.add_program(device_range, std::move(binary_program));
        auto& binary_program_ = binary_workload.get_programs().at(device_range);
        input_a = 3;
        input_b = 4;

        // 2. Compute the hash of the original kernel file, for use in CreateKernelFromBinary.
        auto binary_hash = tt_metal::experimental::ComputeKernelOriginalPathHash(kernel_file);

        // 3a. Set the binary path prefix for the device.
        tt_metal::experimental::SetKernelBinaryPathPrefix(mesh_device.get(), binary_kernel_path);

        // 3b. Call CreateKernelFromBinary to create each kernel from the binaries.
        auto kernel_handle_binary = tt_metal::experimental::CreateKernelFromBinary(
            binary_program_,
            "simple_add",
            binary_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0, .compile_args = {}},
            binary_hash);
        SetRuntimeArgs(binary_program_, kernel_handle_binary, binary_core, {table_address, input_a, input_b});
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), binary_workload, false);
        distributed::Finish(mesh_device->mesh_command_queue());
        tt_metal::detail::ReadFromDeviceL1(device, binary_core, table_address, sizeof(uint32_t), result);
        EXPECT_EQ(result[0], input_a + input_b);

    }
}

}  // namespace tt::tt_metal
