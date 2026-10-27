// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// The branch predictor indexes by a hash of the PC, so a taken jump elsewhere can make it predict an interrupt
// handler's mret as taken, and the core then skips the mret. The kernel's handler retries a skipped mret: every
// interrupt must return, and at least one mret must have been skipped, or the jump did not alias it.
TEST_F(QuasarMeshDeviceSingleCardFixture, TriscMretWithAliasedBranchReturns) {
    // The kernel searches the scratch for an address that aliases the mret; any 128 KB holds one.
    constexpr uint32_t scratch_bytes = 128 * 1024;
    constexpr uint32_t iterations = 1000;
    auto mesh_device = devices_[0];
    const experimental::NodeCoord node{0, 0};

    distributed::DeviceLocalBufferConfig local_cfg{.page_size = scratch_bytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig global_cfg{.size = scratch_bytes};
    auto buffer = distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());
    const uint32_t scratch = buffer->address();
    std::vector<uint32_t> results(4, 0xffffffff);
    slow_dispatch::WriteToL1(*mesh_device, node, scratch, results);

    const experimental::KernelSpecName kernel{"mret_alias"};
    experimental::KernelSpec kernel_spec{
        .unique_id = kernel,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/trisc_mret_btb_alias.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"scratch", "scratch_bytes", "iterations"}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };
    experimental::WorkUnitSpec work_unit{.name = "main", .kernels = {kernel}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = "trisc_mret_alias", .kernels = {kernel_spec}, .work_units = {work_unit}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = kernel,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node, {{"scratch", scratch}, {"scratch_bytes", scratch_bytes}, {"iterations", iterations}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);

    slow_dispatch::ReadFromL1(*mesh_device, node, scratch, results.size() * sizeof(uint32_t), results);
    const uint32_t handled = results[0];
    const uint32_t skipped = results[1];
    const uint32_t unfinished = results[2];
    const uint32_t jump = results[3];
    ASSERT_NE(jump, 0u) << "no scratch address aliases the handler's mret";
    log_info(LogTest, "TRISC0: {} interrupts handled, {} mrets skipped, jump at {:#x}", handled, skipped, jump);
    EXPECT_EQ(unfinished, 0u) << "an interrupt was never taken or never returned";
    EXPECT_EQ(handled, iterations);
    EXPECT_GT(skipped, 0u) << "no mret was skipped: the jump did not alias the handler's mret";
}
