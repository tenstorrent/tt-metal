// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "device_fixture.hpp"
#include "jit_build/build.hpp"  // jit_build_cache_clear

using namespace tt::tt_metal;

namespace {

// Build a single-DM-kernel program. Distinct processors / cores keep the programs distinct so the
// batch exercises more than one compile.
Program make_program(DataMovementProcessor processor, const CoreCoord& core) {
    const std::string kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    Program program = CreateProgram();
    CreateKernel(program, kernel_file, core, DataMovementConfig{.processor = processor});
    return program;
}

}  // namespace

// Compiling a batch warms every program with no errors, and reports the resolved worker count.
TEST_F(MeshDeviceFixture, CompileProgramsBatchWarmsAllProgramsErrorFree) {
    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        jit_build_cache_clear();

        std::vector<Program> programs;
        programs.push_back(make_program(DataMovementProcessor::RISCV_0, CoreCoord(0, 0)));
        programs.push_back(make_program(DataMovementProcessor::RISCV_1, CoreCoord(0, 0)));
        programs.push_back(make_program(DataMovementProcessor::RISCV_0, CoreCoord(1, 0)));

        std::vector<Program*> ptrs;
        ptrs.reserve(programs.size());
        for (auto& p : programs) {
            ptrs.push_back(&p);
        }

        auto stats = detail::CompilePrograms(device, ptrs, /*max_workers=*/4);

        EXPECT_EQ(stats.num_programs, ptrs.size());
        EXPECT_EQ(stats.num_errors, 0u);
        EXPECT_EQ(stats.max_workers, 4);
    }
}

// An empty batch is a no-op, not an error.
TEST_F(MeshDeviceFixture, CompileProgramsEmptyBatchIsNoOp) {
    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];

        std::vector<Program*> none;
        auto stats = detail::CompilePrograms(device, none, /*max_workers=*/4);

        EXPECT_EQ(stats.num_programs, 0u);
        EXPECT_EQ(stats.num_errors, 0u);
    }
}

// max_workers <= 0 resolves to hardware concurrency (>= 1), and the batch still compiles cleanly.
TEST_F(MeshDeviceFixture, CompileProgramsResolvesDefaultWorkerCount) {
    for (const auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        jit_build_cache_clear();

        Program program = make_program(DataMovementProcessor::RISCV_0, CoreCoord(0, 0));
        std::vector<Program*> ptrs{&program};

        auto stats = detail::CompilePrograms(device, ptrs, /*max_workers=*/0);

        EXPECT_EQ(stats.num_programs, 1u);
        EXPECT_EQ(stats.num_errors, 0u);
        EXPECT_GE(stats.max_workers, 1);
    }
}
