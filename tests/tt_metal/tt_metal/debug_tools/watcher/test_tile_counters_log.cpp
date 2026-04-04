// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "impl/context/metal_context.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher tile counter log feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {

// Key idea: detect and read incomplete mismatch between posted vs acked on all 64 TCs
// Setup: 16 DFBs, 1 producer, 4 consumers = 16 x 4 = 64 TCs
// Producer posts tiles, some consumers exit early -> TC mismatches

constexpr uint32_t NUM_DFBS = 16;
constexpr uint32_t NUM_PRODUCERS = 1;
constexpr uint32_t NUM_CONSUMERS = 4;
constexpr uint32_t NUM_ENTRIES_PER_DFB = 16;
constexpr uint32_t NUM_ENTRIES_PER_PRODUCER = NUM_ENTRIES_PER_DFB / NUM_PRODUCERS;
constexpr uint32_t NUM_ENTRIES_PER_CONSUMER = NUM_ENTRIES_PER_DFB / NUM_CONSUMERS;
constexpr uint32_t NUM_CONSUMERS_TO_RUN = 2;              // 2 consumers run, 2 exit early -> 32 TCs mismatch
constexpr uint32_t TOTAL_TCS = NUM_DFBS * NUM_CONSUMERS;  // Use all 64 TCs

void RunTest(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, {});
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_tile_counters.cpp";

    // Allocate and zero-init L1 sync flag for host-device handshake
    uint32_t tensix_sync_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::vector<uint32_t> zero_data = {0};
    tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, tensix_sync_addr, zero_data);

    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    // DFB config: 1 producer, 4 consumers per DFB
    experimental::dfb::DataflowBufferConfig dfb_config{
        .entry_size = 64,
        .num_entries = NUM_ENTRIES_PER_DFB,
        .num_producers = NUM_PRODUCERS,
        .num_consumers = NUM_CONSUMERS,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    std::vector<uint32_t> producer_cta = {NUM_DFBS, NUM_ENTRIES_PER_PRODUCER};
    std::vector<uint32_t> consumer_cta = {NUM_DFBS, NUM_ENTRIES_PER_CONSUMER, NUM_CONSUMERS_TO_RUN};

    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        kernel_path,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = dfb_config.num_producers,
            .compile_args = producer_cta,
            .defines = {{"DFB_PRODUCER", "1"}}});

    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        kernel_path,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = dfb_config.num_consumers, .compile_args = consumer_cta});

    // Create DFBs and bind to producer/consumer kernels
    for (uint32_t i = 0; i < NUM_DFBS; i++) {
        auto dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
        experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program, dfb_id, producer_kernel, consumer_kernel);
    }

    // Set common runtime args: sync_flag_addr
    SetCommonRuntimeArgs(program, producer_kernel, {tensix_sync_addr});
    SetCommonRuntimeArgs(program, consumer_kernel, {tensix_sync_addr});

    // Dispatch w/o blocking: kernels post tiles then spin on sync flag
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, virtual_core);

    // Build Watcher log pattern for mismatched TCs (consumers 2, 3 -> TC indices 32, 33, ... 63)
    std::vector<std::string> expected_mismatch_patterns;
    for (uint32_t dfb = 0; dfb < NUM_DFBS; dfb++) {
        for (uint32_t consumer = NUM_CONSUMERS_TO_RUN; consumer < NUM_CONSUMERS; consumer++) {
            uint32_t tc_idx = consumer * NUM_DFBS + dfb;
            expected_mismatch_patterns.push_back(fmt::format("TC[{}]: posted=", tc_idx));
        }
    }
    log_info(tt::LogTest, "Polling {} TC mismatch patterns...", expected_mismatch_patterns.size());

    constexpr uint32_t timeout_ms = 30000;
    auto start = std::chrono::steady_clock::now();
    // Wait for TC mismatch string to appear
    while (!FileContainsAllStrings(fixture->log_file_name, expected_mismatch_patterns)) {
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        ASSERT_LT(elapsed, timeout_ms) << "Timed out waiting for watcher to log TC mismatches";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    log_info(tt::LogTest, "TC patterns found!");

    // Release sync flag to unblock kernels
    std::vector<uint32_t> release_data = {1};
    tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, tensix_sync_addr, release_data);
    distributed::Finish(mesh_device->mesh_command_queue());
}

TEST_F(MeshWatcherDumpAllFixture, TestWatcherTileCounterLog) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_tile_counter_registers()) {
        GTEST_SKIP() << "Tile counters are only used on Quasar";
    }
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device);
            },
            mesh_device);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
