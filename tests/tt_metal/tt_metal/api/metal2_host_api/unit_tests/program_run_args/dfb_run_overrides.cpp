// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramRunArgs::dfb_run_overrides: DFB size overrides, including aliased DFB groups.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"
#include "metal2_host_api/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeRunArgsForMinimalSpec;
using test_helpers::MakeSpecWithAliasedDfbs;
using test_helpers::MakeSpecWithRTAs;
using test_helpers::ProgramRunArgsTestQuasar;

TEST_F(ProgramRunArgsTestQuasar, CPU_SetRunArgsSucceeds_DFBRunOverridesWithNoOverrides) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});

    // DFB run params with no overrides is allowed
    params.dfb_run_overrides.push_back(ProgramRunArgs::DFBRunOverrides{
        .dfb = DFBSpecName{"dfb_0"},
        // No overrides - both entry_size and num_entries are nullopt
    });

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

// First-launch override: entry_size only (per-TC base/limit recompute; capacity unchanged).
TEST_F(ProgramRunArgsTestQuasar, CPU_DFBEntrySizeOverrideSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .entry_size = 2048});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->config.entry_size, 2048u);
    EXPECT_EQ(dfb->config.num_entries, 2u);  // unchanged
    EXPECT_EQ(dfb->capacity, 2u);            // capacity = num_entries / max(prod,cons)
    EXPECT_EQ(dfb->stride_in_entries, 1u);
}

// First-launch override: num_entries only (changes capacity).
TEST_F(ProgramRunArgsTestQuasar, CPU_DFBNumEntriesOverrideSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 4});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->config.entry_size, 1024u);  // unchanged
    EXPECT_EQ(dfb->config.num_entries, 4u);
    EXPECT_EQ(dfb->capacity, 4u);
    EXPECT_EQ(dfb->stride_in_entries, 1u);
}

// First-launch override: both entry_size and num_entries.
TEST_F(ProgramRunArgsTestQuasar, CPU_DFBBothOverridesSucceed) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .entry_size = 512, .num_entries = 8});

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->config.entry_size, 512u);
    EXPECT_EQ(dfb->config.num_entries, 8u);
    EXPECT_EQ(dfb->capacity, 8u);
}

TEST_F(ProgramRunArgsTestQuasar, CPU_DFBEntrySizeOverrideZeroFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .entry_size = 0});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("entry_size must be set to a non-zero value")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_DFBNumEntriesOverrideZeroFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 0});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("num_entries must be set to a non-zero value")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_DFBSizeOverrideUnknownNameFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"does_not_exist"}, .entry_size = 2048});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Unknown DFB spec name")));
}

TEST_F(ProgramRunArgsTestQuasar, CPU_DuplicateDFBParamsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithRTAs(node, 0, 0);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    // Push dfb_0 twice with no size overrides (so size-override guard does not fire first).
    params.dfb_run_overrides.push_back(ProgramRunArgs::DFBRunOverrides{.dfb = DFBSpecName{"dfb_0"}});
    params.dfb_run_overrides.push_back(ProgramRunArgs::DFBRunOverrides{.dfb = DFBSpecName{"dfb_0"}});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate DFB 'dfb_0'")));
}

// Isolated change on the primary: entry_size and num_entries traded so total_size is unchanged.
TEST_F(ProgramRunArgsTestQuasar, CPU_AliasIsolatedResizeSucceeds) {
    NodeCoord node{0, 0};
    // dfb_a = 512*8 = 4096, dfb_b = 1024*4 = 4096 (equal totals).
    ProgramSpec spec = MakeSpecWithAliasedDfbs(/*es_a=*/512, /*ne_a=*/8, /*es_b=*/1024, /*ne_b=*/4);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto a = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_a"));
    auto b = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_b"));
    ASSERT_TRUE(a->alias_primary_id.has_value() || !a->alias_secondary_ids.empty()) << "dfb_a not aliased";
    const uint32_t total_before = a->total_size();
    const uint32_t b_es_before = b->config.entry_size;
    const uint32_t b_ne_before = b->config.num_entries;

    // 512*8 -> 256*16: total_size stays 4096.
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_a"}, .entry_size = 256, .num_entries = 16});
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));

    EXPECT_EQ(a->config.entry_size, 256u);
    EXPECT_EQ(a->config.num_entries, 16u);
    EXPECT_EQ(a->total_size(), total_before);  // unchanged -> isolated
    // The other alias is untouched.
    EXPECT_EQ(b->config.entry_size, b_es_before);
    EXPECT_EQ(b->config.num_entries, b_ne_before);
}

// Isolated change on the secondary alias.
TEST_F(ProgramRunArgsTestQuasar, CPU_AliasSecondaryIsolatedResizeSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithAliasedDfbs(/*es_a=*/512, /*ne_a=*/8, /*es_b=*/1024, /*ne_b=*/4);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto b = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_b"));
    const uint32_t total_before = b->total_size();

    // 1024*4 -> 2048*2: total_size stays 4096.
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_b"}, .entry_size = 2048, .num_entries = 2});
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));

    EXPECT_EQ(b->config.entry_size, 2048u);
    EXPECT_EQ(b->config.num_entries, 2u);
    EXPECT_EQ(b->total_size(), total_before);  // unchanged -> isolated
}

// Agreed group resize: BOTH members overridden to the same new total size.
TEST_F(ProgramRunArgsTestQuasar, CPU_AliasGroupAgreedResizeSucceeds) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithAliasedDfbs(/*es_a=*/512, /*ne_a=*/8, /*es_b=*/1024, /*ne_b=*/4);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto a = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_a"));
    auto b = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_b"));

    // 4096 -> 8192 for both, via different views (512*16 and 1024*8).
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_a"}, .entry_size = 512, .num_entries = 16});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_b"}, .entry_size = 1024, .num_entries = 8});
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));

    EXPECT_EQ(a->total_size(), 8192u);
    EXPECT_EQ(b->total_size(), 8192u);
    EXPECT_EQ(a->config.num_entries, 16u);
    EXPECT_EQ(b->config.num_entries, 8u);
}

// Total-size change on one alias without overriding the rest of the group -> rejected.
TEST_F(ProgramRunArgsTestQuasar, CPU_AliasPartialGroupResizeFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithAliasedDfbs(/*es_a=*/512, /*ne_a=*/8, /*es_b=*/1024, /*ne_b=*/4);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // dfb_a 4096 -> 8192 (total changes) but dfb_b is left out of the batch.
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_a"}, .num_entries = 16});
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("was not overridden")));
}

// Group members overridden to DIFFERENT new total sizes -> rejected.
TEST_F(ProgramRunArgsTestQuasar, CPU_AliasGroupDisagreeResizeFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeSpecWithAliasedDfbs(/*es_a=*/512, /*ne_a=*/8, /*es_b=*/1024, /*ne_b=*/4);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // dfb_a -> 8192, dfb_b -> 16384: both change total_size but disagree.
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_a"}, .num_entries = 16});  // 512*16 = 8192
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_b"}, .num_entries = 16});  // 1024*16 = 16384
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("different total sizes")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
