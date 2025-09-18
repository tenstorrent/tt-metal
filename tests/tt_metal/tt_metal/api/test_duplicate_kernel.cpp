// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/core_descriptor.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <map>
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
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/utils.hpp>

namespace tt::tt_metal {

using namespace tt;

// Ensures we cannot create duplicate kernels
TEST_F(MeshDispatchFixture, TensixFailOnDuplicateKernelCreationDataflow) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        std::shared_ptr<distributed::MeshDevice> mesh_device = this->devices_.at(id);
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        auto& program_ = workload.get_programs().at(device_range);

        CoreCoord compute_grid = this->devices_.at(id)->compute_with_storage_grid_size();
        EXPECT_THROW(
            {
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            },
            std::exception);
    }
}

TEST_F(MeshDispatchFixture, TensixFailOnDuplicateKernelCreationCompute) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        std::shared_ptr<distributed::MeshDevice> mesh_device = this->devices_.at(id);
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        auto& program_ = workload.get_programs().at(device_range);

        CoreCoord compute_grid = this->devices_.at(id)->compute_with_storage_grid_size();
        std::vector<uint32_t> compute_kernel_args = {};
        EXPECT_THROW(
            {
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/compute/broadcast.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false,
                        .compile_args = compute_kernel_args,
                        .opt_level = KernelBuildOptLevel::O3});
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false,
                        .compile_args = compute_kernel_args,
                        .opt_level = KernelBuildOptLevel::O3});
            },
            std::exception);
    }
}

TEST_F(MeshDispatchFixture, TensixPassOnNormalKernelCreation) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        std::shared_ptr<distributed::MeshDevice> mesh_device = this->devices_.at(id);
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        auto& program_ = workload.get_programs().at(device_range);
        std::vector<uint32_t> compute_kernel_args = {};
        EXPECT_NO_THROW({
            tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/compute/broadcast.cpp",
                CoreCoord(1, 0),
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = false,
                    .math_approx_mode = false,
                    .compile_args = compute_kernel_args,
                    .opt_level = KernelBuildOptLevel::O3});
            tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
                CoreCoord(0, 0),
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = false,
                    .math_approx_mode = false,
                    .compile_args = compute_kernel_args,
                    .opt_level = KernelBuildOptLevel::O3});
        });
    }
}

TEST_F(MeshDispatchFixture, TensixPassOnMixedOverlapKernelCreation) {
    for (unsigned int id = 0; id < this->devices_.size(); id++) {
        std::shared_ptr<distributed::MeshDevice> mesh_device = this->devices_.at(id);
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        auto& program_ = workload.get_programs().at(device_range);
        CoreCoord compute_grid = this->devices_.at(id)->compute_with_storage_grid_size();
        std::vector<uint32_t> compute_kernel_args = {};
        EXPECT_NO_THROW({
            tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
                CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = false,
                    .math_approx_mode = false,
                    .compile_args = compute_kernel_args,
                    .opt_level = KernelBuildOptLevel::O3});
        });
    }
}

}  // namespace tt::tt_metal
