// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/kernel_cores.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>

#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {
namespace {

bool same_cores(const CoreRangeSet& a, const CoreRangeSet& b) { return a.contains(b) && b.contains(a); }

}  // namespace

// GetKernelCores is the union of the cores of a program's kernels of one core type: what dispatch places the program
// by. Cores that hold a circular buffer or semaphore but run no kernel are not among them.
TEST_F(MeshDispatchFixture, TensixKernelCoresAreTheUnionOfTheProgramsKernels) {
    Program program = CreateProgram();
    EXPECT_TRUE(experimental::GetKernelCores(program, HalProgrammableCoreType::TENSIX).empty());

    const CoreRangeSet reader_cores(std::vector{CoreRange({0, 0}, {1, 1}), CoreRange({4, 2}, {4, 3})});
    const CoreRangeSet compute_cores(CoreRange({1, 1}, {2, 1}));
    const CoreRangeSet object_only_cores(CoreRange({6, 0}, {6, 0}));
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        reader_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(program, "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp", compute_cores, ComputeConfig{});
    CreateSemaphore(program, reader_cores.merge(object_only_cores), 0);

    EXPECT_TRUE(same_cores(
        experimental::GetKernelCores(program, HalProgrammableCoreType::TENSIX), reader_cores.merge(compute_cores)));
    EXPECT_TRUE(experimental::GetKernelCores(program, HalProgrammableCoreType::ACTIVE_ETH).empty());
    EXPECT_TRUE(experimental::GetKernelCores(program, HalProgrammableCoreType::IDLE_ETH).empty());
}

}  // namespace tt::tt_metal
