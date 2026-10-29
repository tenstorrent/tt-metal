// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <functional>
#include <stdexcept>
#include <utility>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/sram_arena/sram_arena.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::PrefetcherPipeSpecTestGen1;
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

ProgramSpec MakePipeSenderArenaScratchpadSpecForTest() {
    ProgramSpec spec = MakeArenaScratchpadSpecForTest();
    spec.kernels[0].advanced_options.prefetcher_pipe_bindings = {test_helpers::BindPipe()};
    spec.advanced_options.prefetcher_pipe_parameters = {test_helpers::MakePipeParameter()};
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work", test_helpers::pipe_sender_node, {"writer"})};
    return spec;
}

ProgramRunArgs MakeArenaAndPipeRunArgsForTest(const SRAMArena& arena, PrefetcherPipe& pipe) {
    ProgramRunArgs args = MakeArenaRunArgsForTest(arena);
    args.advanced_options.prefetcher_pipe_args = {{test_helpers::pipe_param_name, std::ref(pipe)}};
    return args;
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaEnqueueRejectsUnboundScratchpad) {
    // Enqueue rejects an arena scratchpad when initial arguments were never supplied.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto program = MakeProgramFromSpec(fixture_device, MakeArenaScratchpadSpecForTest());
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(fixture_device.shape()), std::move(program));
    EXPECT_THAT(
        [&] { distributed::EnqueueMeshWorkload(fixture_device.mesh_command_queue(), workload, false); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no bound SRAMArena")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaEnqueueRejectsMissingScratchpadAfterUncheckedSet) {
    // Skipping argument validation does not allow an unbound arena scratchpad to be dispatched.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto program = MakeProgramFromSpec(fixture_device, MakeArenaScratchpadSpecForTest());
    SetProgramRunArgs(program, {}, /*skip_validation=*/true);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(fixture_device.shape()), std::move(program));
    EXPECT_THAT(
        [&] { distributed::EnqueueMeshWorkload(fixture_device.mesh_command_queue(), workload, false); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no bound SRAMArena")));
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

TEST_F(PrefetcherPipeSpecTestGen1, CPU_SRAMArenaEnqueueRejectsScratchpadAfterSetRejectedByPipe) {
    // A SetProgramRunArgs call rejected by its pipe argument does not bind the arena scratchpad.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, LockstepSRAMArenaProperties{4096});
    PrefetcherPipe other_sender_pipe = MakePipeFor(fixture_device, test_helpers::MakePipeParameter(), CoreCoord{2, 0});
    auto program = MakeProgramFromSpec(fixture_device, MakePipeSenderArenaScratchpadSpecForTest());
    ASSERT_THAT(
        [&] { SetProgramRunArgs(program, MakeArenaAndPipeRunArgsForTest(arena, other_sender_pipe)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("the supplied pipe's sender is (2,0)")));
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(fixture_device.shape()), std::move(program));
    EXPECT_THAT(
        [&] { distributed::EnqueueMeshWorkload(fixture_device.mesh_command_queue(), workload, false); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no bound SRAMArena")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
