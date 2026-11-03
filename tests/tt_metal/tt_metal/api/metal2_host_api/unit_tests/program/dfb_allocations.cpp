// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/overlapping_dfb_allocations.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::ProgramSpecTestBlackhole;

TEST_F(ProgramSpecTestBlackhole, CPU_DFBResizePreservesSharedCoreAllocationSafety) {
    auto program = MakeProgramFromSpec(*mesh_device_, test_helpers::MakeOverlappingDFBAllocationSpec());
    SetProgramRunArgs(program, {});
    for (uint32_t entries : {16u, 2u, 8u}) {
        SCOPED_TRACE(entries);
        ProgramRunArgs args;
        args.dfb_run_overrides.push_back({.dfb = DFBSpecName{"d_b"}, .num_entries = entries});
        UpdateProgramRunArgs(program, args);
        program.impl().prepare_ephemeral_allocation(mesh_device_.get());
        program.impl().allocate_dataflow_buffers(mesh_device_.get());
        const auto a = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("d_a"));
        const auto b = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("d_b"));
        const auto ab = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("d_ab"));
        EXPECT_GE(ab->uniform_alloc_addr(), a->uniform_alloc_addr() + a->total_size());
        EXPECT_GE(ab->uniform_alloc_addr(), b->uniform_alloc_addr() + b->total_size());
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
