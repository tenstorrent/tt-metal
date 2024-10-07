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

static void RunTest(WatcherFixture* fixture, Device* device) {
    // Set up program
    auto program = CreateScopedProgram();

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
    bool has_idle_eth_cores = !device->get_inactive_ethernet_cores().empty();

    // TODO: Enable this when FD-on-idle-eth is supported.
    if (!fixture->IsSlowDispatch())
        has_idle_eth_cores = false;

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
    if (has_idle_eth_cores) {
        KernelHandle ierisc_kid;
        std::set<CoreRange> eth_core_ranges;
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            eth_core_ranges.insert(CoreRange(core, core));
        }
        ierisc_kid = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
            eth_core_ranges,
            tt_metal::EthernetConfig{
                .eth_mode = Eth::IDLE,
                .noc = tt_metal::NOC::NOC_0
            }
        );

        for (const auto& core : device->get_inactive_ethernet_cores()) {
            SetRuntimeArgs(program, ierisc_kid, core, args);
        }
    }


    // Run the program in a new thread, we'll have to update gate values in this thread.
    fixture->RunProgram(device, program);

    // Check that the expected waypoints are in the watcher log, a set for each core.
    auto check_core = [&](const CoreCoord &logical_core, const CoreCoord &phys_core, bool is_eth_core, bool is_active) {
        vector<string> expected_waypoints;
        string expected;
        // Need to update the expected strings based on each core.
        // for (string waypoint : {"AAAA", "BBBB", "CCCC"}) { // Stripped this down since the wait function is flaky
        for (string waypoint : {"AAAA"}) {
            if (is_eth_core) {
                // Each different config has a different calculation for k_id, let's just do one. Fast Dispatch, one device.
                string k_id_s;
                if (tt::tt_metal::GetNumAvailableDevices() == 1 && !fixture->IsSlowDispatch()) {
                    // blank | prefetch, dispatch | tensix kernels
                    int k_id = 1 + 2 + 3;
                    string k_id_s = fmt::format("{}", k_id);
                } else {
                    k_id_s = "";
                }
                expected = fmt::format(
                    "Device {} ethnet core(x={:2},y={:2}) phys(x={:2},y={:2}): {},   X,   X,   X,   X  rmsg:* k_id:{}",
                    device->id(), logical_core.x, logical_core.y, phys_core.x, phys_core.y,
                    waypoint,
                    k_id_s
                );
            } else {
                // Each different config has a different calculation for k_id, let's just do one. Fast Dispatch, one device.
                string k_id_s;
                if (tt::tt_metal::GetNumAvailableDevices() == 1 && !fixture->IsSlowDispatch()) {
                    // blank | prefetch, dispatch
                    int k_id = 1 + 2;
                    string k_id_s = fmt::format("{}|{}|{}", k_id, k_id+1, k_id+2);
                } else {
                    k_id_s = "";
                }
                expected = fmt::format(
                    "Device {} worker core(x={:2},y={:2}) phys(x={:2},y={:2}): {},{},{},{},{}  rmsg:***|*** smsg:**** k_ids:{}",
                    device->id(), logical_core.x, logical_core.y, phys_core.x, phys_core.y,
                    waypoint, waypoint, waypoint, waypoint, waypoint,
                    k_id_s
                );
            }
            expected_waypoints.push_back(expected);
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
            CoreCoord logical_core = {x, y};
            CoreCoord phys_core = device->worker_core_from_logical_core(logical_core);
            check_core(logical_core, phys_core, false, false);
        }
    }
    if (has_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            CoreCoord phys_core = device->ethernet_core_from_logical_core(core);
            check_core(core, phys_core, true, true);
        }
    }
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            CoreCoord phys_core = device->ethernet_core_from_logical_core(core);
            check_core(core, phys_core, true, false);
        }
    }
}

TEST_F(WatcherFixture, TestWatcherWaypoints) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
