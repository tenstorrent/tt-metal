// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include "debug_tools_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

void RunTriscIrqMretAliasTest(
    MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // The kernel searches this for an address that aliases the mret. The predictor hashes PC bits 16:2, so any
    // 128 KB holds one.
    constexpr uint32_t scratch_bytes = 128 * 1024;
    const experimental::NodeCoord node{0, 0};

    distributed::DeviceLocalBufferConfig local_cfg{.page_size = scratch_bytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig global_cfg{.size = scratch_bytes};
    auto buffer = distributed::MeshBuffer::create(global_cfg, local_cfg, mesh_device.get());
    const uint32_t scratch = buffer->address();
    std::vector<uint32_t> results(4, 0);
    slow_dispatch::WriteToL1(*mesh_device, node, scratch, results);

    const experimental::KernelSpecName kernel{"irq_mret_alias"};
    experimental::KernelSpec kernel_spec{
        .unique_id = kernel,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/trisc_irq_mret_alias.cpp",
        .num_threads = 1,
        .runtime_arg_schema = {.runtime_arg_names = {"scratch", "scratch_bytes"}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };
    experimental::WorkUnitSpec work_unit{.name = "main", .kernels = {kernel}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = "trisc_irq_mret_alias", .kernels = {kernel_spec}, .work_units = {work_unit}};
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = kernel,
        .runtime_arg_values =
            experimental::MakeRuntimeArgsForSingleNode(node, {{"scratch", scratch}, {"scratch_bytes", scratch_bytes}}),
    }};
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    fixture->RunProgramExpectingWatcherError(mesh_device, workload);

    std::string exception;
    constexpr auto timeout = std::chrono::milliseconds(10000);
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        if (!exception.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    slow_dispatch::ReadFromL1(*mesh_device, node, scratch, results.size() * sizeof(uint32_t), results);
    ASSERT_EQ(results[0], 1u) << "TRISC1 never finished preparing";
    ASSERT_NE(results[2], 0u) << "no mret found in the firmware error handler";
    ASSERT_NE(results[3], 0u) << "no scratch address aliases the handler's mret 0x" << std::hex << results[2];
    ASSERT_FALSE(exception.empty()) << "no watcher error within " << timeout.count() << " ms";
    EXPECT_NE(exception.find("ILLEGAL_INSTRUCTION_TRISC0"), std::string::npos) << exception;

    const uint32_t count_after_error = results[1];
    std::this_thread::sleep_for(std::chrono::seconds(1));
    slow_dispatch::ReadFromL1(*mesh_device, node, scratch, results.size() * sizeof(uint32_t), results);
    log_info(
        LogTest,
        "TRISC1 count {} -> {}, mret at {:#x}, jump at {:#x}",
        count_after_error,
        results[1],
        results[2],
        results[3]);
    EXPECT_GT(results[1], count_after_error) << "TRISC1 stopped counting: it did not return from the error handler";
}

}  // namespace

// Checks that a TRISC returns from the HW error handler when another branch aliases its mret.
TEST_F(MeshWatcherFixture, QuasarTriscReturnsFromErrorHandlerWithAliasedMret) {
    if (MetalContext::instance().hal().get_arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar only";
    }
    this->RunTestOnDevice(RunTriscIrqMretAliasTest, this->devices_[0]);
}
