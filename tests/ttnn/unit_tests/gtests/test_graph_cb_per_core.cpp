// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>
#include "gtest/gtest.h"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::graph::test {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

class CBPerCoreGraphTestFixture : public TTNNFixtureWithSuiteDevice<CBPerCoreGraphTestFixture> {};

// A program placing circular buffers on DISJOINT core sets. No single core holds all of them, so the
// per-core L1 peak must be the busiest core's total, not the sum over every CB in the program.
TEST_F(CBPerCoreGraphTestFixture, PeakTotalIsPerCoreAcrossDisjointCoreRangeSets) {
    constexpr uint32_t kTile = 32 * 32 * 2;
    constexpr uint32_t kSizeOuter = 4 * kTile;   // cores x=0 and x=3
    constexpr uint32_t kSizeInner = 8 * kTile;   // cores x=1..2
    constexpr uint32_t kSizeShared = 2 * kTile;  // all four cores

    const CoreRangeSet all_cores(CoreRange({0, 0}, {3, 0}));
    const CoreRangeSet outer_cores(std::set<CoreRange>{CoreRange({0, 0}, {0, 0}), CoreRange({3, 0}, {3, 0})});
    const CoreRangeSet inner_cores(CoreRange({1, 0}, {2, 0}));

    auto make_cb = [](uint32_t index, uint32_t size) {
        return tt::tt_metal::CircularBufferConfig(size, {{index, tt::DataFormat::Float16_b}})
            .set_page_size(index, kTile);
    };

    auto program = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateCircularBuffer(program, outer_cores, make_cb(0, kSizeOuter));
    tt::tt_metal::CreateCircularBuffer(program, inner_cores, make_cb(1, kSizeInner));
    tt::tt_metal::CreateCircularBuffer(program, all_cores, make_cb(2, kSizeShared));
    tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(device_->shape()), std::move(program));

    GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NORMAL);
    tt::tt_metal::distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, /*blocking=*/true);
    tt::tt_metal::distributed::Finish(device_->mesh_command_queue());
    const auto trace = GraphProcessor::end_graph_capture();

    // The trace carries each CB's core_range_set as structured JSON that round-trips through the
    // CoreRangeSet (de)serializer.
    uint32_t cb_nodes = 0;
    for (const auto& node : trace) {
        if (node.at(kNodeType) != kNodeCBAllocate) {
            continue;
        }
        cb_nodes++;
        const auto& core_range_set_json = node.at(kParams).at(kCoreRangeSet);
        ASSERT_TRUE(core_range_set_json.is_array()) << core_range_set_json.dump();
        const auto core_range_set = ttsl::json::from_json<CoreRangeSet>(core_range_set_json);
        const auto size = node.at(kParams).at(kSize).get<uint32_t>();
        if (size == kSizeOuter) {
            EXPECT_EQ(core_range_set, outer_cores);
        } else if (size == kSizeInner) {
            EXPECT_EQ(core_range_set, inner_cores);
        } else {
            EXPECT_EQ(size, kSizeShared);
            EXPECT_EQ(core_range_set, all_cores);
        }
    }
    EXPECT_EQ(cb_nodes, 3u);

    // The busiest core (x=1 or x=2) holds the inner CB and the shared CB; the outer CB never shares a
    // core with the inner one, so it must not be added on top.
    const auto usage = extract_resource_usage_per_core(trace);
    EXPECT_EQ(usage.peak_cb, kSizeInner + kSizeShared);
    EXPECT_EQ(usage.peak_total, kSizeInner + kSizeShared);
    EXPECT_EQ(usage.peak_l1, 0u);
}

}  // namespace ttnn::graph::test
