// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramRunArgs::global_semaphore_args: binding GlobalSemaphore objects to a Program's
// GlobalSemaphoreParameters, and rebinding a different one on Update.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/global_semaphore.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>

#include "impl/buffers/global_semaphore_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/global_semaphore_test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::GlobalSemaphoreArgs;
using test_helpers::MakeGlobalSemaphoreSpec;
using test_helpers::ProgramRunArgsTestGen1;

#define EXPECT_GLOBAL_SEM_THROWS_WITH(stmt, substr) \
    EXPECT_THAT([&] { stmt; }, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(substr)))

const NodeRange kGlobalSemNodes{NodeCoord{0, 0}, NodeCoord{1, 1}};

uint32_t GlobalSemTableEntry(const Program& program, const std::string& name) {
    const auto& impl = program.impl();
    return impl.global_semaphore_table().at(1 + impl.get_global_semaphore_parameter(name)->slot);
}

// GlobalSemaphores are created before the programs that use them: placing program-local L1 on a
// core seals that core's persistent arena. Tests below follow that order.

TEST_F(ProgramRunArgsTestGen1, CPU_PersistentSemaphoresGetDistinctArenaAddresses) {
    GlobalSemaphore a = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    GlobalSemaphore b = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    EXPECT_TRUE(a.impl().is_persistent());
    EXPECT_TRUE(b.impl().is_persistent());
    EXPECT_NE(a.address(), b.address());
}

TEST_F(ProgramRunArgsTestGen1, CPU_SetRunArgsWritesAddressIntoTable) {
    GlobalSemaphore sem = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, kGlobalSemNodes));

    SetProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &sem}}));

    EXPECT_TRUE(program.impl().get_global_semaphore_parameter("gsem")->bound);
    EXPECT_EQ(GlobalSemTableEntry(program, "gsem"), static_cast<uint32_t>(sem.address()));
}

TEST_F(ProgramRunArgsTestGen1, CPU_TwoParametersGetTheirOwnSlots) {
    GlobalSemaphore first = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    GlobalSemaphore second = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"first", "second"}, kGlobalSemNodes));

    SetProgramRunArgs(program, GlobalSemaphoreArgs({{"first", &first}, {"second", &second}}));

    EXPECT_EQ(GlobalSemTableEntry(program, "first"), static_cast<uint32_t>(first.address()));
    EXPECT_EQ(GlobalSemTableEntry(program, "second"), static_cast<uint32_t>(second.address()));
}

// A semaphore placed on more cores than the program uses is fine.
TEST_F(ProgramRunArgsTestGen1, CPU_SemaphoreOnSupersetOfNodesIsAccepted) {
    const NodeRange superset{NodeCoord{0, 0}, NodeCoord{3, 3}};
    GlobalSemaphore sem = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(superset), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, kGlobalSemNodes));

    EXPECT_NO_THROW(SetProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &sem}})));
}

// Unlike PrefetcherPipe, a different GlobalSemaphore can be bound on Update; only the table changes.
TEST_F(ProgramRunArgsTestGen1, CPU_UpdateRebindsDifferentSemaphore) {
    GlobalSemaphore first = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    GlobalSemaphore second = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, kGlobalSemNodes));

    SetProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &first}}));
    UpdateProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &second}}));

    EXPECT_EQ(GlobalSemTableEntry(program, "gsem"), static_cast<uint32_t>(second.address()));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateMayOmitParameterAndKeepsPriorAddress) {
    GlobalSemaphore first = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    GlobalSemaphore second = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"first", "second"}, kGlobalSemNodes));

    SetProgramRunArgs(program, GlobalSemaphoreArgs({{"first", &first}, {"second", &second}}));
    UpdateProgramRunArgs(program, GlobalSemaphoreArgs({{"second", &first}}));

    EXPECT_EQ(GlobalSemTableEntry(program, "first"), static_cast<uint32_t>(first.address()));
    EXPECT_EQ(GlobalSemTableEntry(program, "second"), static_cast<uint32_t>(first.address()));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SetRunArgsMissingParameterIsRejected) {
    GlobalSemaphore first = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"first", "second"}, kGlobalSemNodes));

    EXPECT_GLOBAL_SEM_THROWS_WITH(
        SetProgramRunArgs(program, GlobalSemaphoreArgs({{"first", &first}})),
        "GlobalSemaphoreParameter 'second' is declared in the Program but has no GlobalSemaphoreArgument entry");
}

TEST_F(ProgramRunArgsTestGen1, CPU_UnknownParameterIsRejected) {
    GlobalSemaphore sem = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, kGlobalSemNodes));

    EXPECT_GLOBAL_SEM_THROWS_WITH(
        SetProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &sem}, {"other", &sem}})),
        "the Program declares no GlobalSemaphoreParameter of that name");
}

// A buffer-backed GlobalSemaphore has a per-core address only by allocator luck; the table needs one.
TEST_F(ProgramRunArgsTestGen1, CPU_NonPersistentSemaphoreIsRejected) {
    GlobalSemaphore sem = tt::tt_metal::CreateGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, kGlobalSemNodes));

    EXPECT_GLOBAL_SEM_THROWS_WITH(
        SetProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &sem}})), "is not a persistent GlobalSemaphore");
}

TEST_F(ProgramRunArgsTestGen1, CPU_SemaphoreNotCoveringBinderNodesIsRejected) {
    GlobalSemaphore sem =
        CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, kGlobalSemNodes));

    EXPECT_GLOBAL_SEM_THROWS_WITH(
        SetProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &sem}})), "which do not cover the nodes");
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateWithNonPersistentSemaphoreIsRejected) {
    GlobalSemaphore persistent = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    GlobalSemaphore buffer_backed =
        tt::tt_metal::CreateGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeGlobalSemaphoreSpec({"gsem"}, kGlobalSemNodes));
    SetProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &persistent}}));

    EXPECT_GLOBAL_SEM_THROWS_WITH(
        UpdateProgramRunArgs(program, GlobalSemaphoreArgs({{"gsem", &buffer_backed}})),
        "is not a persistent GlobalSemaphore");
    EXPECT_EQ(GlobalSemTableEntry(program, "gsem"), static_cast<uint32_t>(persistent.address()));
}

TEST_F(ProgramRunArgsTestGen1, CPU_MergeRejectsSameParameterTwice) {
    GlobalSemaphore sem = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    std::vector<ProgramRunArgs> rest{GlobalSemaphoreArgs({{"gsem", &sem}})};

    EXPECT_GLOBAL_SEM_THROWS_WITH(
        MergeProgramRunArgs(GlobalSemaphoreArgs({{"gsem", &sem}}), rest),
        "GlobalSemaphoreParameter 'gsem' is specified in more than one ProgramRunArgs");
}

TEST_F(ProgramRunArgsTestGen1, CPU_MergeUnionsDisjointParameters) {
    GlobalSemaphore first = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    GlobalSemaphore second = CreatePersistentGlobalSemaphore(*mesh_device_, NodeRangeSet(kGlobalSemNodes), 0);
    std::vector<ProgramRunArgs> rest{GlobalSemaphoreArgs({{"second", &second}})};

    ProgramRunArgs merged = MergeProgramRunArgs(GlobalSemaphoreArgs({{"first", &first}}), rest);
    EXPECT_EQ(merged.global_semaphore_args.size(), 2u);
}

#undef EXPECT_GLOBAL_SEM_THROWS_WITH

}  // namespace
}  // namespace tt::tt_metal::experimental
