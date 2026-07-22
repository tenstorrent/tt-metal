// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Watcher test for the device-zero "zero -> barrier -> reuse" contract guard
// (tt_metal/hw/inc/internal/debug/noc_zero_guard.h). A kernel that issues a NoC write between
// Noc::async_write_zeros() and Noc::write_zeros_l1_barrier() must trip NOC_ASSERT_NOT_ZERO_MODE();
// the same kernel with the barrier in place must run clean. The guard is Quasar-specific (only
// Quasar borrows the write command buffer for the L1 zero), so this test runs on Quasar only.
//
// Uses the Metal 2.0 host API (KernelSpec/ProgramSpec + DataflowBuffer), mirroring
// tests/tt_metal/tt_metal/api/test_zero_memory_api.cpp, plus the MeshWatcherFixture exception-poll
// pattern from tests/tt_metal/tt_metal/debug_tools/watcher/test_assert.cpp. A DFB needs its producer
// and consumer on different RISCs, so the violation lives in the producer and a small consumer
// drains the DFB; on the violation path only the producer hangs (it signals dispatch-done first so
// Finish() returns) and the consumer early-returns.

#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "hal_types.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "debug_tools_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

const experimental::DFBSpecName SCRATCH_DFB{"scratch"};
const experimental::KernelSpecName PRODUCER{"zero_mode_producer"};
const experimental::KernelSpecName CONSUMER{"zero_mode_consumer"};
constexpr uint32_t kScratchBytes = 8 * 1024;
constexpr uint32_t kZeroBytes = 2 * 1024;

experimental::DataMovementHardwareConfig make_hw_config() {
    return experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
}

Program make_program(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const experimental::NodeCoord node{0, 0};

    experimental::DataflowBufferSpec scratch_spec{
        .unique_id = SCRATCH_DFB,
        .entry_size = kScratchBytes,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Producer issues the L1 zero and (only on the violation path) a NoC write before the barrier.
    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source =
            std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/watcher_zero_mode_violation.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::PRODUCER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .runtime_arg_schema = {.runtime_arg_names = {"should_trip", "zero_bytes"}},
        .hw_config = make_hw_config(),
    };

    // Consumer drains the DFB (different RISC than the producer, as DFBs require).
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/dataflow/zero_memory_api_consumer.cpp"},
        .num_threads = 1,
        .dfb_bindings =
            {{.dfb_spec_name = SCRATCH_DFB,
              .accessor_name = "scratch",
              .endpoint_type = experimental::DFBEndpointType::CONSUMER,
              .access_pattern = experimental::DFBAccessPattern::STRIDED}},
        .hw_config = make_hw_config(),
    };

    experimental::ProgramSpec spec{
        .name = "watcher_zero_mode_guard",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {scratch_spec},
        .work_units = {{.name = "main", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node}},
    };
    return experimental::MakeProgramFromSpec(*mesh_device, spec);
}

void set_args(Program& program, uint32_t should_trip) {
    const experimental::NodeCoord node{0, 0};
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"should_trip", should_trip}, {"zero_bytes", kZeroBytes}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);
}

void RunZeroModeTest(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    distributed::MeshCoordinateRange device_range(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    workload.add_program(device_range, make_program(mesh_device));
    auto& program = workload.get_programs().at(device_range);

    // 1) Safe run: barrier present -> the guard must not fire.
    set_args(program, /*should_trip=*/0);
    fixture->RunProgram(mesh_device, workload, /*wait_for_dump=*/true);
    EXPECT_TRUE(MetalContext::instance().watcher_server()->exception_message().empty())
        << "Safe run (barrier present) unexpectedly tripped the watcher: "
        << MetalContext::instance().watcher_server()->exception_message();

    // 2) Violation run: NoC write with no intervening barrier -> watcher assert (the producer signals
    // dispatch-done before hanging, so RunProgram's Finish() returns).
    set_args(program, /*should_trip=*/1);
    fixture->RunProgram(mesh_device, workload);

    std::string exception;
    constexpr auto timeout = std::chrono::milliseconds(5000);
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        if (!exception.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_FALSE(exception.empty()) << "Timeout waiting for the watcher to catch the zero-mode contract violation.";
    EXPECT_TRUE(exception.find("tripped an assert") != std::string::npos)
        << "Watcher exception was not an assert: " << exception;
    EXPECT_TRUE(exception.find("watcher_zero_mode_violation") != std::string::npos)
        << "Watcher assert did not name the zero-mode kernel: " << exception;
}

}  // namespace

TEST_F(MeshWatcherFixture, TensixTestWatcherZeroModeContract) {
    const bool is_quasar = MetalContext::instance().hal().get_arch() == tt::ARCH::QUASAR;
    if (!is_quasar) {
        GTEST_SKIP() << "The device-zero command-buffer guard is Quasar-specific";
    }
    this->RunTestOnDevice(
        [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunZeroModeTest(fixture, mesh_device);
        },
        this->devices_[0]);
}
