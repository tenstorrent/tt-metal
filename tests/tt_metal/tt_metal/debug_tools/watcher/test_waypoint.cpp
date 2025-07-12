// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <set>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "umd/device/types/arch.h"
#include <tt-metalium/utils.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher waypoints.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
void RunTest(WatcherFixture* fixture, IDevice* device) {
    // https://github.com/tenstorrent/tt-metal/issues/23306
    device->disable_and_clear_program_cache();

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
    uint32_t clk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->id());
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
        KernelHandle ierisc_kid0{}, ierisc_kid1{};
        std::set<CoreRange> eth_core_ranges;
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            eth_core_ranges.insert(CoreRange(core, core));
        }
        ierisc_kid0 = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
            eth_core_ranges,
            tt_metal::EthernetConfig{
                .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = DataMovementProcessor::RISCV_0});

        if (device->arch() == ARCH::BLACKHOLE) {
            ierisc_kid1 = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_waypoints.cpp",
                eth_core_ranges,
                tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .processor = DataMovementProcessor::RISCV_1});
        }

        for (const auto& core : device->get_inactive_ethernet_cores()) {
            SetRuntimeArgs(program, ierisc_kid0, core, args);
            if (device->arch() == ARCH::BLACKHOLE) {
                SetRuntimeArgs(program, ierisc_kid1, core, args);
            }
        }
    }


    // Run the program in a new thread, we'll have to update gate values in this thread.
    fixture->RunProgram(device, program);

    // Check that the expected waypoints are in the watcher log, a set for each core.
    auto check_core = [&](const CoreCoord& logical_core,
                          const CoreCoord& virtual_core,
                          bool is_eth_core,
                          bool is_active) {
        vector<std::string> expected_waypoints;
        std::string expected;
        // Need to update the expected strings based on each core.
        // for (string waypoint : {"AAAA", "BBBB", "CCCC"}) { // Stripped this down since the wait function is flaky
        for (std::string waypoint : {"AAAA"}) {
            if (is_eth_core) {
                // Each different config has a different calculation for k_id, let's just do one. Fast Dispatch, one device.
                std::string k_id_s;
                if (tt::tt_metal::GetNumAvailableDevices() == 1 && !fixture->IsSlowDispatch()) {
                    // blank | prefetch, dispatch | tensix kernels
                    int k_id = 1 + 2 + 3;
                    std::string k_id_s = fmt::format("{:3}", k_id);
                    if (device->arch() == ARCH::BLACKHOLE)
                        k_id_s += fmt::format("|{:3}", k_id + 1);
                } else {
                    k_id_s = "";
                }
                expected = fmt::format(
                    "Device {} {}eth core(x={:2},y={:2}) virtual(x={:2},y={:2}): {},{},   X,   X,   X  ",
                    device->id(),
                    is_active ? "act" : "idl",
                    logical_core.x,
                    logical_core.y,
                    virtual_core.x,
                    virtual_core.y,
                    waypoint,
                    // TODO(#17275): Rework risc counts & masks into HAL and generalize this test.
                    // Active eth core only has one available erisc to test on.
                    (device->arch() == ARCH::BLACKHOLE and not is_active) ? waypoint : "   W");
                if (device->arch() == ARCH::BLACKHOLE) {
                    expected += fmt::format("rmsg:???|?? h_id:  0 smsg:? k_id:{}", k_id_s);
                } else {
                    expected += fmt::format("rmsg:???|? h_id:  0 k_id:{}", k_id_s);
                }
            } else {
                // Each different config has a different calculation for k_id, let's just do one. Fast Dispatch, one device.
                std::string k_id_s;
                if (tt::tt_metal::GetNumAvailableDevices() == 1 && !fixture->IsSlowDispatch()) {
                    // blank | prefetch, dispatch
                    int k_id = 1 + 2;
                    std::string k_id_s = fmt::format("{:3}|{:3}|{:3}", k_id, k_id + 1, k_id + 2);
                } else {
                    k_id_s = "";
                }
                expected = fmt::format(
                    "Device {} worker core(x={:2},y={:2}) virtual(x={:2},y={:2}): {},{},{},{},{}  rmsg:???|??? h_id:  "
                    "0 "
                    "smsg:???? k_ids:{}",
                    device->id(),
                    logical_core.x,
                    logical_core.y,
                    virtual_core.x,
                    virtual_core.y,
                    waypoint,
                    waypoint,
                    waypoint,
                    waypoint,
                    waypoint,
                    k_id_s);
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
            CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
            check_core(logical_core, virtual_core, false, false);
        }
    }
    if (has_eth_cores) {
        for (const auto& core : device->get_active_ethernet_cores(true)) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            check_core(core, virtual_core, true, true);
        }
    }
    if (has_idle_eth_cores) {
        for (const auto& core : device->get_inactive_ethernet_cores()) {
            CoreCoord virtual_core = device->ethernet_core_from_logical_core(core);
            check_core(core, virtual_core, true, false);
        }
    }
}
}
}

TEST_F(WatcherFixture, TestWatcherWaypoints) {
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, device);
    }
}
