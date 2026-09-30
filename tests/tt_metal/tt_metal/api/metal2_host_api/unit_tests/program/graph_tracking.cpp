// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dataflow buffers and scratchpads that a Program reports to graph tracking.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/distributed.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeBorrowedDFBProgramSpecForRunArgs;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeSpecWithAliasedDfbs;
using test_helpers::ProgramRunArgsTestQuasar;

// ============================================================================
// Dataflow buffers reported to graph tracking
// ============================================================================
//
// A hooked program never allocates, so GraphTracker reports its DFBs as CB allocations (#51674).
// The alias skip, borrowed flag and hook gate each stop over-reporting, so each gets a test.

// Records what a capture would see, so the reporting can be checked without a TTNN graph capture.
class RecordingGraphProcessor : public IGraphProcessor {
public:
    struct DfbAllocation {
        uint64_t size = 0;
        bool borrows_memory = false;
    };

    void track_allocate_cb(
        const CoreRangeSet& /*core_range_set*/,
        uint64_t /*addr*/,
        uint64_t size,
        bool /*is_globally_allocated*/,
        const IDevice* /*device*/) override {
        cb_sizes.push_back(size);
    }

    void track_allocate_dataflow_buffer(
        const CoreRangeSet& /*core_range_set*/,
        uint64_t /*addr*/,
        uint64_t size,
        bool borrows_memory,
        const IDevice* /*device*/) override {
        dfb_allocations.push_back(DfbAllocation{.size = size, .borrows_memory = borrows_memory});
    }

    void track_allocate_scratchpad(
        const CoreRangeSet& /*core_range_set*/, uint64_t /*addr*/, uint64_t size, const IDevice* /*device*/) override {
        scratchpad_sizes.push_back(size);
    }

    std::vector<uint64_t> cb_sizes;
    std::vector<DfbAllocation> dfb_allocations;
    std::vector<uint64_t> scratchpad_sizes;
};

// Blocking hooks stand in for RunMode::NO_DISPATCH, where programs are captured but never run.
class BlockingGraphHooks : public IGraphHooks {
public:
    bool hook_allocate(const Buffer*) override { return true; }
    bool hook_deallocate(Buffer*) override { return true; }
    bool hook_program(Program*) override { return true; }
    bool hook_write_to_device(const Buffer*) override { return true; }
    bool hook_write_to_device(const distributed::MeshBuffer*) override { return true; }
    bool hook_read_from_device(Buffer*) override { return true; }
    bool hook_read_from_device(const distributed::MeshBuffer*) override { return true; }
};

class ScopedGraphTracking {
public:
    ScopedGraphTracking(const std::shared_ptr<IGraphProcessor>& processor, bool block_programs) {
        GraphTracker::instance().push_processor(processor);
        if (block_programs) {
            GraphTracker::instance().add_hook(std::make_shared<BlockingGraphHooks>());
        }
    }
    ~ScopedGraphTracking() {
        GraphTracker::instance().pop_processor();
        GraphTracker::instance().clear_hook();
    }
};

TEST_F(ProgramRunArgsTestQuasar, CPU_TrackProgramCollapsesAliasedDataflowBuffers) {
    // Equal totals (512*8 == 1024*4) so the assertion holds whichever member becomes primary.
    ProgramSpec spec = MakeSpecWithAliasedDfbs(/*es_a=*/512, /*ne_a=*/8, /*es_b=*/1024, /*ne_b=*/4);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Guards against passing vacuously if aliasing silently failed to apply.
    auto a = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_a"));
    ASSERT_TRUE(a->alias_primary_id.has_value() || !a->alias_secondary_ids.empty()) << "dfb_a not aliased";

    auto processor = std::make_shared<RecordingGraphProcessor>();
    {
        ScopedGraphTracking tracking(processor, /*block_programs=*/true);
        GraphTracker::instance().track_program(&program, mesh_device_.get());
    }

    ASSERT_EQ(processor->dfb_allocations.size(), 1u) << "aliased DFBs share one L1 region, report it once";
    EXPECT_EQ(processor->dfb_allocations[0].size, 4096u);
    EXPECT_FALSE(processor->dfb_allocations[0].borrows_memory);
    EXPECT_TRUE(processor->cb_sizes.empty()) << "a dataflow buffer is not a circular buffer";
}

TEST_F(ProgramRunArgsTestQuasar, CPU_TrackProgramFlagsBorrowedDataflowBuffer) {
    // entry_size 16 * num_entries 2 = 32 bytes.
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunArgs();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto processor = std::make_shared<RecordingGraphProcessor>();
    {
        ScopedGraphTracking tracking(processor, /*block_programs=*/true);
        GraphTracker::instance().track_program(&program, mesh_device_.get());
    }

    ASSERT_EQ(processor->dfb_allocations.size(), 1u);
    EXPECT_TRUE(processor->dfb_allocations[0].borrows_memory)
        << "a borrowed DFB is backed by a tensor that is tracked in its own right";
    EXPECT_EQ(processor->dfb_allocations[0].size, 32u);
}

// Scratchpads are program-scope L1 stacked on top of the DFB region, so a consumer summing L1 has
// to see them too or it under-reports — the wrong direction for a "does this fit" query.
TEST_F(ProgramRunArgsTestQuasar, CPU_TrackProgramReportsKernelScratchpads) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // one DFB, entry_size 1024 * num_entries 2
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"}};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto processor = std::make_shared<RecordingGraphProcessor>();
    {
        ScopedGraphTracking tracking(processor, /*block_programs=*/true);
        GraphTracker::instance().track_program(&program, mesh_device_.get());
    }

    ASSERT_EQ(processor->dfb_allocations.size(), 1u);
    EXPECT_EQ(processor->dfb_allocations[0].size, 2048u);
    EXPECT_THAT(processor->scratchpad_sizes, ::testing::ElementsAre(1024u));
    EXPECT_TRUE(processor->cb_sizes.empty());
}

// Without a hook the program goes on to run, and its allocations report themselves with real
// addresses. Reporting here too would double-count them.
TEST_F(ProgramRunArgsTestQuasar, CPU_TrackProgramSkipsDataflowBuffersWhenProgramIsNotHooked) {
    ProgramSpec spec = MakeSpecWithAliasedDfbs(/*es_a=*/512, /*ne_a=*/8, /*es_b=*/1024, /*ne_b=*/4);
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto processor = std::make_shared<RecordingGraphProcessor>();
    {
        ScopedGraphTracking tracking(processor, /*block_programs=*/false);
        GraphTracker::instance().track_program(&program, mesh_device_.get());
    }

    EXPECT_TRUE(processor->dfb_allocations.empty());
    EXPECT_TRUE(processor->scratchpad_sizes.empty());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
