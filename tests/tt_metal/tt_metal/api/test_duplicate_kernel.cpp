// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/vocabulary/core_coord.hpp>
#include "llrt/core_descriptor.hpp"
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "compile_program_with_kernel_path_env_var_fixture.hpp"
#include <tt-metalium/workload/legacy/kernel_types.hpp>
#include <tt-metalium/mesh/legacy/device.hpp>
#include <tt-metalium/mesh/dispatch_core_config.hpp>
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/execution/distributed.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/workload/program.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/xy_pair.hpp>

namespace tt::tt_metal {

using namespace tt;

// Ensures we cannot create duplicate kernels
TEST_F(MeshDispatchFixture, TensixFailOnDuplicateKernelCreationDataflow) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Not ported to Quasar: creates a DataMovementKernel, which Quasar rejects";
    }
    for (const auto& device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram(*device);
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        CoreCoord compute_grid = device->compute_with_storage_grid_size();
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
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Not ported to Quasar: creates a ComputeKernel, which Quasar rejects";
    }
    for (const auto& device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram(*device);
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        CoreCoord compute_grid = device->compute_with_storage_grid_size();
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
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Not ported to Quasar: creates a ComputeKernel, which Quasar rejects";
    }
    for ([[maybe_unused]] const auto& mesh_device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram(*mesh_device);
        workload.add_program(device_range, std::move(program));
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
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Not ported to Quasar: creates DataMovement and Compute kernels, which Quasar rejects";
    }
    for (const auto& device : this->devices_) {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram(*device);
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);
        CoreCoord compute_grid = device->compute_with_storage_grid_size();
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
