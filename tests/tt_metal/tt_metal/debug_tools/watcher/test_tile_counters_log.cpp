// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt_stl/assert.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hal_types.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking debug ring buffer feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

void RunTest(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, {});
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    uint32_t total_dfbs = 16;  // 16 x 4 consumers = 64 TCs

    experimental::dfb::DataflowBufferConfig config {
        .entry_size = 1024;
        .num_entries = 16;
        .num_producers = 1;
        .pap = AccessPattern::STRIDED;
        .num_consumers = 4;
        .enable_implicit_sync = false
    };

    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, virtual_core);

    // Run the program
    fixture->RunProgram(mesh_device, workload, true);

    log_info(tt::LogTest, "Checking file: {}", fixture->log_file_name);

    // Check log
    EXPECT_TRUE(FileContainsAllStringsInOrder(fixture->log_file_name, expected));
}

TEST_F(MeshWatcherFixture, TestWatcherTileCounterLog) {
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
