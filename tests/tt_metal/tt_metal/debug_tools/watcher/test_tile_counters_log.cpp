// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>
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
// Tests DM -> 4 NEO unpacker transfers: 1 DM producer, 4 NEO TRISC consumers
// Parameterized to test both strided mode (bypass) and blocked mode (remapper enabled)
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {

// Setup: 1 DFB with 1 DM producer -> 4 NEO unpacker consumers
// Producer posts tiles, some consumers exit early -> TC mismatches
constexpr uint32_t NUM_PRODUCERS = 1;
constexpr uint32_t NUM_CONSUMERS = 4;  // 4 NEO unpackers
constexpr uint32_t NUM_ENTRIES_PER_DFB = 16;
constexpr uint32_t NUM_ENTRIES_PER_PRODUCER = NUM_ENTRIES_PER_DFB / NUM_PRODUCERS;
constexpr uint32_t NUM_ENTRIES_PER_CONSUMER = NUM_ENTRIES_PER_DFB / NUM_CONSUMERS;
constexpr uint32_t NUM_CONSUMERS_TO_RUN = 2;  // 2 NEOs consume, 2 exit early -> TC mismatch
constexpr uint32_t TILE_SIZE = 2048;          // Single tile size for compute

void RunTest(
    MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool use_remapper) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, {});
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);

    // Allocate L1 buffer for sync flag
    tt_metal::InterleavedBufferConfig sync_buffer_config{
        .device = device,
        .size = sizeof(uint32_t),
        .page_size = sizeof(uint32_t),
        .buffer_type = tt_metal::BufferType::L1};
    auto sync_buffer = CreateBuffer(sync_buffer_config);
    uint32_t tensix_sync_addr = sync_buffer->address();
    std::vector<uint32_t> zero_data = {0};
    tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, tensix_sync_addr, zero_data);

    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    // DFB config: 1 DM producer -> 4 NEO unpacker consumers
    // use_remapper: true -> all (remapper enabled), false -> strided (bypass mode)
    dfb::AccessPattern cap = use_remapper ? dfb::AccessPattern::ALL : dfb::AccessPattern::STRIDED;
    experimental::dfb::DataflowBufferConfig dfb_config{
        .entry_size = TILE_SIZE,
        .num_entries = NUM_ENTRIES_PER_DFB,
        .num_producers = NUM_PRODUCERS,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = NUM_CONSUMERS,
        .cap = cap,
        .enable_implicit_sync = false,
        .data_format = tt::DataFormat::Float16_b};

    auto dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_range_set, dfb_config);
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_tile_counters.cpp";

    // DM producer kernel
    std::vector<uint32_t> producer_cta = {dfb_id, NUM_ENTRIES_PER_PRODUCER};
    auto producer_kernel = experimental::quasar::CreateKernel(
        program,
        kernel_path,
        core_range_set,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = NUM_PRODUCERS,
            .compile_args = producer_cta,
            .defines = {{"DFB_PRODUCER", "1"}}});

    // NEO compute consumer kernel (4 threads = 4 Neo clusters)
    // blocked: each consumer sees all entries; strided: entries distributed among consumers
    uint32_t entries_per_consumer = use_remapper ? NUM_ENTRIES_PER_DFB : NUM_ENTRIES_PER_CONSUMER;
    // Compile args: dfb_id, num_entries, num_consumers_to_run, sync_flag_addr
    std::vector<uint32_t> consumer_cta = {dfb_id, entries_per_consumer, NUM_CONSUMERS_TO_RUN, tensix_sync_addr};
    auto consumer_kernel = experimental::quasar::CreateKernel(
        program,
        kernel_path,
        core_range_set,
        experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = NUM_CONSUMERS, .compile_args = consumer_cta});

    experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, dfb_id, producer_kernel, consumer_kernel);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    std::string mode_name = use_remapper ? "remapper" : "bypass";
    log_info(
        LogTest,
        "Running DM->NEO test ({}) on device {} core {}[{}]...",
        mode_name,
        device->id(),
        logical_core,
        virtual_core);

    // Build expected patterns: early exiting consumers (2, 3) don't ack, creating mismatches
    // tiles_to_consume = posted - acked; since acked = 0 for early-exit consumers, tiles_to_consume =
    // entries_per_consumer
    std::vector<std::string> expected_mismatch_patterns;
    for (uint32_t consumer = NUM_CONSUMERS_TO_RUN; consumer < NUM_CONSUMERS; consumer++) {
        if (use_remapper) {
            // Remapper: tiles_to_consume shows unprocessed tiles per consumer
            expected_mismatch_patterns.push_back(
                fmt::format("-> NEO_{} tc_id:0 tiles_to_consume:{}", consumer, entries_per_consumer));
        } else {
            // Bypass: each consumer TC shows tiles_to_consume
            expected_mismatch_patterns.push_back(
                fmt::format("remapper:N/A NEO_{} tc_id:0 tiles_to_consume:{}", consumer, entries_per_consumer));
        }
    }
    log_info(tt::LogTest, "Polling {} TC mismatch patterns ({})...", expected_mismatch_patterns.size(), mode_name);

    auto release_threads = [&]() {
        std::vector<uint32_t> release_data = {1};
        tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, tensix_sync_addr, release_data);
        distributed::Finish(mesh_device->mesh_command_queue());
    };

    constexpr uint32_t timeout_ms = 30000;
    constexpr uint32_t poll_interval_ms = 1000;
    auto start = std::chrono::steady_clock::now();

    while (!FileContainsAllStrings(fixture->log_file_name, expected_mismatch_patterns)) {
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed_ms >= timeout_ms) {
            release_threads();
            FAIL() << "Timed out after " << timeout_ms << "ms waiting for watcher to log TC mismatches (" << mode_name
                   << ")";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }

    log_info(tt::LogTest, "TC patterns found! ({})", mode_name);
    release_threads();
}

// Test bypass mode (strided consumer access pattern)
TEST_F(MeshWatcherDumpAllFixture, TestWatcherTileCounterLogBypass) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_tile_counter_registers()) {
        GTEST_SKIP() << "Tile counters are only used on Quasar";
    }
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, /* use_remapper= */ false);
            },
            mesh_device);
    }
}

// Test remapper mode (blocked consumer access pattern)
TEST_F(MeshWatcherDumpAllFixture, TestWatcherTileCounterLogRemapper) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_tile_counter_registers()) {
        GTEST_SKIP() << "Tile counters are only used on Quasar";
    }
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, /* use_remapper= */ true);
            },
            mesh_device);
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
