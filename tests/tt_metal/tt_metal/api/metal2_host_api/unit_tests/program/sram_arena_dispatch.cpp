// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <functional>
#include <utility>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/sram_arena/sram_arena.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::ProgramRunArgsTestGen1;

ProgramSpec MakeArenaScratchpadSpecForTest() {
    ProgramSpec spec;
    spec.kernels = {test_helpers::MakeMinimalGen1DMKernel("writer")};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 1024}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = 1024};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = LockstepSRAMArenaProperties{4096}}};
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})};
    return spec;
}

ProgramRunArgs MakeArenaRunArgsForTest(const SRAMArena& arena) {
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    return args;
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaEnqueueAcceptsBoundScratchpad) {
    // An arena scratchpad can be dispatched after its arena argument is supplied.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeArenaScratchpadSpecForTest());
    SetProgramRunArgs(program, MakeArenaRunArgsForTest(arena));
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(fixture_device.shape()), std::move(program));
    EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(fixture_device.mesh_command_queue(), workload, false));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaEnqueueAcceptsScratchpadAfterOmittedArenaUpdate) {
    // A partial update can omit an arena that is already bound before dispatch.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeArenaScratchpadSpecForTest());
    SetProgramRunArgs(program, MakeArenaRunArgsForTest(arena));
    UpdateProgramRunArgs(program, {});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(fixture_device.shape()), std::move(program));
    EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(fixture_device.mesh_command_queue(), workload, false));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaEnqueueAllowsLocalScratchpadWithoutRunArgs) {
    // A locally allocated scratchpad with no user arguments can be dispatched without setting arguments.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaScratchpadSpecForTest();
    spec.scratchpads[0].advanced_options.allocation_override.reset();
    spec.advanced_options.sram_arena_parameters.clear();
    auto program = MakeProgramFromSpec(fixture_device, spec);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(fixture_device.shape()), std::move(program));
    EXPECT_NO_THROW(distributed::EnqueueMeshWorkload(fixture_device.mesh_command_queue(), workload, false));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
