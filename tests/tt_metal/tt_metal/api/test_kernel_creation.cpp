// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <set>
#include <string>
#include <variant>
#include <vector>

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
#include <tt-metalium/utils.hpp>

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
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
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

TEST_F(MeshDispatchFixture, DISABLED_TensixCreateKernelsOnStorageCores) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        auto mesh_device = this->devices_.at(id);
        if (mesh_device->storage_only_cores().empty()) {
            GTEST_SKIP() << "This test only runs on devices with storage only cores";
        }

        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        auto& program_ = workload.get_programs().at(device_range);

        const std::set<CoreCoord>& storage_only_cores = mesh_device->storage_only_cores();
        std::set<CoreRange> storage_only_core_ranges;
        for (CoreCoord core : storage_only_cores) {
            storage_only_core_ranges.emplace(core);
        }
        CoreRangeSet storage_core_range_set(storage_only_core_ranges);
        EXPECT_ANY_THROW(
            tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                storage_core_range_set,
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
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
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

}  // namespace tt::tt_metal
