// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "command_queue_fixture.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/overlapping_dfb_allocations.hpp"

namespace tt::tt_metal {
namespace {

TEST_F(UnitMeshCQSingleCardFixture, DFBAllocationsPreservePatternsOnSharedCore) {
    auto& device = *devices_.at(0);
    if (device.arch() != tt::ARCH::WORMHOLE_B0 && device.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "This layout uses Gen1 DM kernels";
    }
    auto program =
        experimental::MakeProgramFromSpec(device, experimental::test_helpers::MakeOverlappingDFBAllocationSpec());
    experimental::SetProgramRunArgs(program, {});
    // Keep the host objects after moving the Program into the workload. Dispatch
    // compiles/configures the actual DFB placement; the no-op kernels leave it idle.
    const auto dfbs = program.impl().dataflow_buffers();
    ASSERT_EQ(dfbs.size(), 3u);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, true);

    // Write every allocation before reading any back. In #62451, d_ab overwrites
    // the upper half of d_b on core (1,0). No FIFO traffic or timing is involved.
    for (size_t i = 0; i < dfbs.size(); ++i) {
        const auto& dfb = dfbs[i];
        std::vector<uint32_t> pattern(dfb->total_size() / sizeof(uint32_t), 0x62451000u + i);
        for (const auto& range : dfb->core_ranges.ranges()) {
            for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    slow_dispatch::WriteToL1(device, {x, y}, dfb->uniform_alloc_addr(), pattern);
                }
            }
        }
    }
    for (size_t i = 0; i < dfbs.size(); ++i) {
        const auto& dfb = dfbs[i];
        const std::vector<uint32_t> expected(dfb->total_size() / sizeof(uint32_t), 0x62451000u + i);
        for (const auto& range : dfb->core_ranges.ranges()) {
            for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    std::vector<uint32_t> actual;
                    slow_dispatch::ReadFromL1(device, {x, y}, dfb->uniform_alloc_addr(), dfb->total_size(), actual);
                    EXPECT_EQ(actual, expected) << "DFB " << i << " on core (" << x << ", " << y << ")";
                }
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal
