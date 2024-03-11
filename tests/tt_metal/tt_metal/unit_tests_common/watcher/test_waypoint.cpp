// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_fixture.hpp"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher waypoints.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

// Some machines will run this test on different physical cores, so wildcard the exact coordinates
// and replace them at runtime.
std::vector<string> ordered_waypoints = {
    "Device *, Core (x=*,y=*):    AAAA,AAAA,AAAA,AAAA,AAAA",
    "Device *, Core (x=*,y=*):    BBBB,BBBB,BBBB,BBBB,BBBB",
    "Device *, Core (x=*,y=*):    CCCC,CCCC,CCCC,CCCC,CCCC"
};
std::vector<string> ordered_waypoints_eth = {
    "Device *, Core (x=*,y=*):    AAAA",
    "Device *, Core (x=*,y=*):    BBBB",
    "Device *, Core (x=*,y=*):    CCCC"
};

static void RunTest(WatcherFixture* fixture, Device* device) {
    // Set up program
    Program program = Program();

    // Test runs on a 5x5 grid
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = {4, 4};

    // Run a kernel that posts waypoints and waits on certain gating values to be written before
    // posting the next waypoint.
    auto brisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto ncrisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto trisc_kid = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
        CoreRange(xy_start, xy_end),
        ComputeConfig{});

    // The kernels need arguments to be passed in: the number of cycles to delay while syncing,
    // and an L1 buffer to use for the syncing.
    uint32_t clk_mhz = tt::Cluster::instance().get_device_aiclk(device->id());
    uint32_t delay_cycles = clk_mhz * 2000000; // 2 seconds
    tt_metal::InterleavedBufferConfig l1_config {
        .device = device,
        .size = sizeof(uint32_t),
        .page_size = sizeof(uint32_t),
        .buffer_type = tt_metal::BufferType::L1
    };
    auto l1_buffer = CreateBuffer(l1_config);

    // Write runtime args
    const std::vector<uint32_t> args = { delay_cycles, l1_buffer->address() };
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            SetRuntimeArgs(
                program,
                brisc_kid,
                CoreCoord{x, y},
                args
            );
            SetRuntimeArgs(
                program,
                ncrisc_kid,
                CoreCoord{x, y},
                args
            );
            SetRuntimeArgs(
                program,
                trisc_kid,
                CoreCoord{x, y},
                args
            );
        }
    }
    // Also run on ethernet cores if they're present
    bool has_eth_cores = !device->get_active_ethernet_cores(true).empty();
    if (has_eth_cores) {
        KernelHandle erisc_kid;
        std::set<CoreRange> eth_core_ranges;
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            eth_core_ranges.insert(CoreRange(core, core));
        }
        erisc_kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
            eth_core_ranges,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});

        for (const auto& core : device->get_active_ethernet_cores(true)) {
            SetRuntimeArgs(program, erisc_kid, core, args);
        }
    }


    // Run the program in a new thread, we'll have to update gate values in this thread.
    fixture->RunProgram(device, program);

    // Check that the expected waypoints are in the watcher log, a set for each core.
    auto check_core = [&](const CoreCoord &phys_core, bool is_eth_core) {
        auto expected_waypoints = (is_eth_core)? ordered_waypoints_eth : ordered_waypoints;
        // Need to update the expected strings based on each core.
        for (int idx = 0; idx < expected_waypoints.size(); idx++) {
            expected_waypoints[idx][7] = '0' + device->id();
            expected_waypoints[idx][18] = '0' + phys_core.x;
            expected_waypoints[idx][22] = '0' + phys_core.y;
        }
        EXPECT_TRUE(
            FileContainsAllStringsInOrder(
                fixture->log_file_name,
                expected_waypoints
            )
        );
    };
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            CoreCoord phys_core = device->worker_core_from_logical_core({x, y});
            check_core(phys_core, false);
        }
    }
    for (const auto& core : device->get_active_ethernet_cores(true)) {
        CoreCoord phys_core = device->ethernet_core_from_logical_core(core);
        check_core(phys_core, true);
    }
}

TEST_F(WatcherFixture, TestWatcherWaypoints) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
